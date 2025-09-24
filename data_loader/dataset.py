"""Dataset definition and loader.

To make it easier to implement datasets into the framework, there is a baseclass
for datasets, called `BaseDynamicalDataset`. This class implements the
`torch.utils.data.Dataset` interface. Datasets suitable for the framework have
the following:
* A temporal component, and the states and observations are 1D tensors.
    The latter means that images, videos, etc. are not suitable. A trajectory
    (state/observation) is of shape (T, D), where T is the number of time steps
    and D is the dimension.
* A linear state to observation mapping.

Minimal example of implementing a dataset:

```
class NewDataset(BaseDynamicalDataset):

    def __init__(self):
        super().__init__(
            num_samples=100,
            signal_length=10,
            smnr_db=10,
            H=torch.eye(2)
        )

    @property
    def observation_noise_covariance(self):
        return torch.eye(2) * 0.1

    def _generate_data(self):
        states = torch.ones(self._num_samples, self._signal_length, 2)
        observations = self.state_to_observation(states)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(2),
            self.observation_noise_covariance,
        )
        noisy_observations = observations + mvn.sample(
            (self._num_samples, self._signal_length)
        )
        return states, observations, noisy_observations
```
"""
import abc
import dataclasses
import glob
from typing import Sequence

import gin
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.data.dataset

import data_loader.linear_ssm as linear_ssm
import data_loader.lorenz_attractor as lorenz_attractor
import data_loader.lorenz96_attractor as lorenz96_attractor


#-----------------------------------------------------
# Functions/classes for datasets in training.
#-----------------------------------------------------

_MEAN = torch.tensor([[[-2.6999e-01,  2.9442e-02,  1.0417e-01, -3.1035e-01, -5.8116e-02,
          -6.6206e-02,  2.5934e-01,  1.6071e-02,  1.4159e-03,  4.7429e-01,
          -7.0643e-02, -9.0259e-02,  5.1831e-01,  3.0504e-03,  7.5386e-02,
          -5.7257e-03, -3.4171e-02,  1.1058e-02, -3.8202e-02,  1.2987e-01,
           8.6504e-05, -1.0689e-01, -1.7201e-01,  8.7006e-02,  4.0161e-02,
          -1.3149e-02,  1.3732e-02, -2.0454e-01,  1.6205e-01,  5.0419e-02,
          -2.1575e-02,  5.3324e-02, -1.2391e-01,  4.0191e-02, -2.1589e-02,
           5.4511e-03, -2.2297e-02, -1.3103e-01, -2.1390e-01, -4.6411e-02,
           1.7103e-01,  1.7414e-01,  2.4868e-02, -4.3447e-02,  3.6690e-02,
           8.3287e-02, -3.2142e-01, -6.9534e-01,  8.9758e-02,  3.0815e-01,
           6.6181e-01,  2.2278e-01, -6.9396e-01,  7.1829e-02,  1.0240e-01,
           8.3668e-01, -1.4374e-01, -3.8368e-02, -7.9196e-02,  8.8692e-02,
          -5.8105e-02,  6.6156e-02, -1.2084e-01]]])
_STD = torch.tensor([[[0.4346, 0.0941, 0.1176, 0.4796, 0.0880, 0.1177, 0.3208, 0.0578,
          0.0547, 0.6605, 0.1558, 0.1876, 0.6485, 0.1658, 0.1569, 0.1258,
          0.0470, 0.0599, 0.1947, 0.1152, 0.1468, 0.2117, 0.1184, 0.1431,
          0.0521, 0.0381, 0.0345, 0.1288, 0.1015, 0.1565, 0.1147, 0.1186,
          0.1597, 0.1428, 0.1381, 0.0777, 0.1054, 0.2248, 0.1753, 0.1091,
          0.2219, 0.1940, 0.1775, 0.1406, 0.0782, 0.1935, 0.2452, 0.3198,
          0.2552, 0.2433, 0.3336, 0.2368, 0.5524, 0.1917, 0.2491, 0.6462,
          0.2690, 0.2877, 0.1029, 0.2553, 0.3147, 0.0999, 0.2919]]])


def _check_same_shape_and_expected_dim(
    *arrays: np.ndarray | torch.Tensor,
    expected_dim: int
) -> None:
    """Checks that all arrays have the same shape and `expected_dim` dimensions.

    Args:
        arrays: The arrays to check.
        expected_dim: The expected number of dimensions.

    Raises:
        ValueError: If the arrays do not have the same shape or number of dimensions.
    """
    first_shape = arrays[0].shape
    for array in arrays:
        if array.shape != first_shape:
            raise ValueError(
                f"Arrays have different shapes: {first_shape} vs {array.shape}"
            )
        if len(array.shape) != expected_dim:
            raise ValueError(
                f"Array has {len(array.shape)} dimensions, expected {expected_dim}"
            )

def _convert_arrays_to_tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    """Converts numpy arrays to PyTorch tensors.
    
    Args:
        arrays: The arrays to convert.
    
    Returns:
        The arrays converted to PyTorch tensors with dtype `torch.float32`.
    """
    return [torch.from_numpy(array).to(dtype=torch.float32) for array in arrays]


class BaseDynamicalDataset(abc.ABC, torch.utils.data.dataset.Dataset):
    """Abstract base class for dynamical system datasets."""

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        H: torch.Tensor,
    ) -> None:
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
            H: State to observation matrix.
        """
        self._num_samples = num_samples
        self._signal_length = signal_length
        self._smnr_db = smnr_db
        self._H = H
        self._states, self._observations, self._noisy_observations, self._Cws = self._generate_data()
        # Expected dimension is 3: (num_samples, signal_length, dim).
        _check_same_shape_and_expected_dim(
            self._states,
            self._observations,
            self._noisy_observations,
            expected_dim=3
        )

    @abc.abstractmethod
    def _generate_data(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the dataset.

        Returns:
            Tuple of three tensors representing 'states', 'observations', and 'noisy observations'.
        """
        pass

    @property
    def H(self) -> torch.Tensor:
        """Returns the observation matrix."""
        return self._H

    def state_to_observation(
        self,
        states: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Maps states to observations by `H @ states`.

        Args:
            stats: The states to map.
            device: The device to use for the conversion, defaults to 'cpu'.

        Returns:
            The observations.
        """
        return torch.einsum(
            '...c,dc->...d',
            states.to(device),
            self._H.to(device)
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int):
        return self._states[index], self._observations[index], self._noisy_observations[index], self._Cws[index]

    def collate_fn(self, batch: Sequence) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
    
        states, observations, noisy_observations, Cws = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        observations = torch.FloatTensor(np.array(observations))
        noisy_observations = torch.FloatTensor(np.array(noisy_observations))
        Cws = torch.FloatTensor(np.array(Cws))
        return states, observations, noisy_observations, Cws


@gin.configurable
class LorenzAttractor(BaseDynamicalDataset):
    """Lorenz attractor dataset."""

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._lorenz_attractor_model = lorenz_attractor.LorenzSSM(delta_d=0.02)
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(3),
        )
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        all_states = []
        noisy_observations = []
        Cws = []
        for _ in range(self._num_samples):
            states, observations = self._lorenz_attractor_model.generate_single_sequence(
                self._signal_length, sigma_e2_dB=-10, smnr_dB=self._smnr_db,
            )
            Cw = np.expand_dims(
                self._lorenz_attractor_model.observation_cov,
                axis=0,
            )
            Cws.append(Cw)
            all_states.append(states)
            noisy_observations.append(observations)
        batched_states = np.stack(all_states, axis=0)
        observations = batched_states.copy()
        noisy_observations = np.stack(noisy_observations, axis=0)
        Cws = np.stack(Cws, axis=0)
        return _convert_arrays_to_tensors(
            batched_states,
            observations,
            noisy_observations,
            Cws,
        )


@gin.configurable
class ChenAttractor(BaseDynamicalDataset):
    """Chen attractor dataset."""

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._chen_attractor_model = lorenz_attractor.LorenzSSM(
            alpha=1.0, delta=0.002, delta_d=0.002 / 5, decimate=True,
        )
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(3),
        )
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        all_states = []
        noisy_observations = []
        Cws = []
        for _ in range(self._num_samples):
            states, observations = self._chen_attractor_model.generate_single_sequence(
                self._signal_length, sigma_e2_dB=-10, smnr_dB=self._smnr_db,
            )
            Cw = np.expand_dims(
                self._chen_attractor_model.observation_cov,
                axis=0,
            )
            Cws.append(Cw)
            all_states.append(states)
            noisy_observations.append(observations)
        batched_states = np.stack(all_states, axis=0)
        observations = batched_states.copy()
        noisy_observations = np.stack(noisy_observations, axis=0)
        Cws = np.stack(Cws, axis=0)
        return _convert_arrays_to_tensors(
            batched_states,
            observations,
            noisy_observations,
            Cws,
        )


@gin.configurable
class Lorenz96Attractor(BaseDynamicalDataset):
    """Lorenz96 attractor dataset."""

    def __init__(
        self,
        dimension: int,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        K: int = 1,
    ) -> None:
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._lorenz96_attractor_model = lorenz96_attractor.Lorenz96SSM(
            n_states=dimension,
            n_obs=dimension,
            delta=0.01,
            delta_d=0.005,
            decimate=False,
            method='RK45',
            F_mu=8.0,
        )
        self._K = K
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(dimension),
        )
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        all_states = []
        noisy_observations = []
        Cws = []
        for _ in range(self._num_samples):
            states, observations = self._lorenz96_attractor_model.generate_single_sequence(
                self._signal_length,
                sigma_e2_dB=-10,  # - 10 * np.log10(self._K),
                smnr_dB=self._smnr_db,
                K=self._K
            )
            Cw = np.expand_dims(
                self._lorenz96_attractor_model.Cw,
                axis=0,
            )
            Cws.append(Cw)
            all_states.append(states)
            noisy_observations.append(observations)
        batched_states = np.stack(all_states, axis=0)
        observations = batched_states.copy()
        noisy_observations = np.stack(noisy_observations, axis=0)
        Cws = np.stack(Cws, axis=0)
        return _convert_arrays_to_tensors(
            batched_states,
            observations,
            noisy_observations,
            Cws
        )


@gin.configurable
class LinearStateSpaceModel(BaseDynamicalDataset):
    """Linear SSM dataset."""

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float) -> None:
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
            smnr_db: The Signal Measurement Noise Ratio in dB.
        """
        self._linear_ssm_model = linear_ssm.LinearSSM()
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.from_numpy(self._linear_ssm_model.H).to(dtype=torch.float32),
        )
        self._Cw = torch.from_numpy(self._linear_ssm_model.Cw).to(dtype=torch.float32)

    @property
    def observation_noise_covariance(self) -> torch.Tensor:
        return self._Cw

    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the double pendulum dataset."""
        all_states = []
        noisy_observations = []
        for _ in range(self._num_samples):
            states, observations = self._linear_ssm_model.generate_single_sequence(
                self._signal_length, sigma_e2_dB=-10, smnr_dB=self._smnr_db,
            )
            all_states.append(states)
            noisy_observations.append(observations)
        batched_states = np.stack(all_states, axis=0)
        observations = self.state_to_observation(
            torch.from_numpy(batched_states).to(dtype=torch.float32)
        ).numpy()
        noisy_observations = np.stack(noisy_observations, axis=0)
        return _convert_arrays_to_tensors(batched_states, observations, noisy_observations)


@gin.configurable
class ClimateData(BaseDynamicalDataset):
    """Climate data from embassy in Beijing."""

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        data_path: str = 'climate_data/data.csv'
    ):
        """Constructor.

        Args:
            signal_length: Number of time steps in the signal.
            observation_noise_db: Observation noise in decibels.
        """
        self._data_path = data_path
        self._data_columns = ["DEWP", "TEMP", "PRES"]
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            H=torch.eye(len(self._data_columns)),
        )
        # We overwrite the number of samples.
        self._num_samples = self._noisy_observations.shape[0]

    @property
    def observation_noise_covariance(self) -> np.ndarray:
        return torch.eye(3) * 10 ** (-self._smnr_db / 10)
    
    def _generate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the climate data."""
        climate_data = pd.read_csv(self._data_path)
        climate_data["datetime"] = pd.to_datetime(climate_data[["year", "month", "day", "hour"]])
        climate_data = climate_data.set_index("datetime").sort_index()
        climate_data_clean = climate_data.dropna(subset=self._data_columns)
        climate_array = np.stack([climate_data_clean[col] for col in self._data_columns], axis=1)
        # Normalize.
        mean = np.mean(climate_array, axis=0)
        std = np.std(climate_array, axis=0)
        climate_array = (climate_array - mean) / std

        full_chunks = len(climate_data) // self._signal_length
        climate_splitted = np.array_split(climate_array[:full_chunks * self._signal_length], full_chunks)
        # We use all data except last, this is used for testing.
        climate_batched = np.stack(climate_splitted[:17] + climate_splitted[31:], axis=0)
        noisy_observations = climate_batched + np.random.multivariate_normal(
            mean=np.zeros(3), cov=self.observation_noise_covariance.numpy(), size=climate_batched.shape[:-1]
        )
        return _convert_arrays_to_tensors(climate_batched, climate_batched, noisy_observations)


@gin.configurable
class BMLmoviDataset(BaseDynamicalDataset):
    """Dataset class for BMLmovi pose data."""
    
    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        base_path: str = "/mimer/NOBACKUP/groups/naiss2025-22-438/BMLmovi"
    ):
        # For BMLmovi, observations are the same as states (identity mapping)
        # Assuming pose_body has shape (time, features), we'll use identity I.
        self.base_path = base_path
        
        sample_data = self._load_sample_data()
        dim = sample_data.shape[1]
        H = torch.eye(dim)
        
        super().__init__(num_samples, signal_length, smnr_db, H)
    
    def _load_sample_data(self) -> np.ndarray:
        """Load a sample file to determine data dimensions."""
        npz_files = glob.glob(f"{self.base_path}/**/*.npz", recursive=True)
        
        for file_path in npz_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'pose_body' in data.files:
                    return data['pose_body']
            except:
                continue
        
        raise ValueError("No valid pose_body data found in NPZ files.")
    
    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and process BMLmovi data."""
        npz_files = glob.glob(f"{self.base_path}/**/*.npz", recursive=True)
        training_subjects = [f'Subject_{i}' for i in range(1, 80)]
        npz_files = [
            f for f in npz_files if any(subj in f for subj in training_subjects)
        ]
        
        all_sequences = []
        
        # Collect all sequences
        for file_path in npz_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'pose_body' in data.files:
                    pose_data = data['pose_body']
                    for i in range(
                        0,
                        len(pose_data) - self._signal_length + 1,
                        self._signal_length
                    ):
                        sequence = pose_data[i:i + self._signal_length]
                        if len(sequence) == self._signal_length:
                            all_sequences.append(sequence)
                            
                        
                        if len(all_sequences) >= self._num_samples:
                            break
                    
                    if len(all_sequences) >= self._num_samples:
                        break
                        
            except Exception as e:
                continue
        
        if len(all_sequences) < self._num_samples:
            print(f"Warning: Only found {len(all_sequences)} sequences, requested {self._num_samples}")
            self._num_samples = len(all_sequences)
        
        states = torch.FloatTensor(all_sequences[:self._num_samples])
        states = (states - _MEAN) / _STD
        
        observations = states.clone()
        dim = states.shape[-1]
        Cws = torch.zeros(self._num_samples, dim, dim)
        noisy_observations = observations.clone()
        
        smnr_linear = 10 ** (self._smnr_db / 10)

        for i in range(self._num_samples):
            signal_power = torch.var(states[i], dim=0)
            noise_power = signal_power / smnr_linear
            min_variance = 1e-3  # or 1e-2 for more stability
            noise_power = torch.clamp(noise_power, min=min_variance)            
            Cws[i] = torch.diag(noise_power)
            
            noise_dist = torch.distributions.MultivariateNormal(
                torch.zeros(dim), 
                Cws[i]
            )
            noise = noise_dist.sample((self._signal_length,))
            noisy_observations[i] += noise
        
        return states, observations, noisy_observations, Cws.unsqueeze(1)



#-----------------------------------------------------
# Functions/classes for datasets in testing.
#-----------------------------------------------------


@dataclasses.dataclass
class MarkovianStateDistribution:
    """Container for Markovian state distribution; contains p(x_t | x_{t-1}).
    
    The distribution is assumed to be Gaussian.
    
    Attributes:
        prev_state: The previous state.
        next_state: The next state; realization of
            N(`mean_next_state`, `cov_next_state`).
        mean_next_state: The mean of the next state.
        cov_next_state: The covariance of the next state.
    """
    
    prev_state: torch.Tensor
    next_state: torch.Tensor
    mean_next_state: torch.Tensor | None
    cov_next_state: torch.Tensor | None


class BaseDynamicalTestDataset(abc.ABC, torch.utils.data.dataset.Dataset):
    """Abstract base class for dynamical system test datasets."""

    def __init__(self) -> None:
        self._markovian_state_distributions = self._generate_markovian_state_distributions()
        self._num_samples = len(self._markovian_state_distributions)

    @abc.abstractmethod
    def _generate_markovian_state_distributions(
        self
    ) -> list[MarkovianStateDistribution]:
        """Generates the dataset."""
        pass

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int):
        current_distribution = self._markovian_state_distributions[index]
        prev_state = current_distribution.prev_state
        next_state = current_distribution.next_state
        mean_next_state = current_distribution.mean_next_state
        cov_next_state = current_distribution.cov_next_state
        return prev_state, next_state, mean_next_state, cov_next_state

    def collate_fn(
        self, batch: Sequence
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
    
        prev_state, next_state, mean_next_state, cov_next_state = zip(*batch)
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            else:
                return torch.from_numpy(x).float()
        
        prev_state = torch.stack([to_tensor(x) for x in prev_state])
        next_state = torch.stack([to_tensor(x) for x in next_state])
        
        if mean_next_state[0] is not None:
            mean_next_state = torch.stack([to_tensor(x) for x in mean_next_state])
            cov_next_state = torch.stack([to_tensor(x) for x in cov_next_state])
        else:
            mean_next_state = None
            cov_next_state = None
            
        return prev_state, next_state, mean_next_state, cov_next_state


@gin.configurable
class LorenzAttractorTest(BaseDynamicalTestDataset):
    """Lorenz attractor dataset."""

    def __init__(self, num_samples: int, signal_length: int):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
        """
        self._lorenz_attractor_model = lorenz_attractor.LorenzSSM(delta_d=0.02)
        self._signal_length = signal_length
        self._num_samples = num_samples
        super().__init__()
    
    def _generate_markovian_state_distributions(
        self
    ) -> list[MarkovianStateDistribution]:
        """Generate the Lorenz attractor test dataset."""
        all_distributions = []
        effective_num_samples = self._num_samples // self._signal_length
        for _ in range(effective_num_samples):
            states = self._lorenz_attractor_model.generate_state_sequence(
                self._signal_length, sigma_e2_dB=-10,
            )
            for t in range(1, self._signal_length):
                prev_state = states[t - 1]
                next_state = states[t]
                mean_next_state = self._lorenz_attractor_model.f_linearize(
                    prev_state
                )
                cov_next_state = self._lorenz_attractor_model.Ce
                all_distributions.append(
                    MarkovianStateDistribution(
                        prev_state=prev_state,
                        next_state=next_state,
                        mean_next_state=mean_next_state,
                        cov_next_state=cov_next_state,
                    )
                )
        return all_distributions


@gin.configurable
class Lorenz96AttractorTest(BaseDynamicalTestDataset):
    """Lorenz attractor dataset."""

    def __init__(
        self,
        dimension: int,
        num_samples: int,
        signal_length: int,
        K: int = 1
    ):
        """Constructor.

        Args:
            num_samples: Number of samples to generate in the dataset.
            signal_length: Number of time steps in the signal.
        """
        self._lorenz96_attractor_model = lorenz96_attractor.Lorenz96SSM(
            n_states=dimension,
            n_obs=dimension,
            delta=0.01,
            delta_d=0.005,
            decimate=False,
            method='RK45',
            F_mu=8.0,
        )
        self._signal_length = signal_length
        self._K = K
        self._num_samples = num_samples
        super().__init__()
    
    def _generate_markovian_state_distributions(
        self
    ) -> list[MarkovianStateDistribution]:
        """Generate the Lorenz attractor test dataset."""
        all_distributions = []
        effective_num_samples = self._num_samples // self._signal_length
        for _ in range(effective_num_samples):
            states = self._lorenz96_attractor_model.generate_state_sequence(
                self._signal_length, sigma_e2_dB=-10,
            )
            for t in range(1, self._signal_length):
                prev_state = states[t - 1]
                next_state = states[t]
                mean_next_state, cov_next_state = self._lorenz96_attractor_model.get_next_state_distribution(
                    prev_state, K=self._K
                )
                mean_next_state = mean_next_state.squeeze()
                cov_next_state = cov_next_state.squeeze()
                all_distributions.append(
                    MarkovianStateDistribution(
                        prev_state=prev_state,
                        next_state=next_state,
                        mean_next_state=mean_next_state,
                        cov_next_state=cov_next_state,
                    )
                )
        return all_distributions


@gin.configurable
class BMLmoviTestDataset(BaseDynamicalTestDataset):
    """Test dataset class for BMLmovi pose data."""
    
    def __init__(
        self,
        num_samples: int,
        base_path: str = "/mimer/NOBACKUP/groups/naiss2025-22-438/BMLmovi"
    ):
        self.base_path = base_path
        self._num_samples = num_samples
        super().__init__()
    
    def _generate_markovian_state_distributions(self) -> list[MarkovianStateDistribution]:
        """Generate Markovian state distributions from BMLmovi data."""
        npz_files = glob.glob(f"{self.base_path}/**/*.npz", recursive=True)
        
        # Filter for test subjects (80+)
        npz_files = [f for f in npz_files if any(f'Subject_{i}' in f for i in range(80, 200))]
        
        distributions = []
        
        for file_path in npz_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'pose_body' in data.files:
                    pose_data = data['pose_body']
                    pose_data = (pose_data - _MEAN.squeeze(0).numpy()) / _STD.squeeze(0).numpy()
                    
                    # Create state pairs (prev_state, next_state)
                    for i in range(len(pose_data) - 1):
                        prev_state = torch.FloatTensor(pose_data[i])
                        next_state = torch.FloatTensor(pose_data[i + 1])
                        
                        # For this dataset, we don't have analytical mean/cov
                        # Set to None (will be handled by collate_fn)
                        distribution = MarkovianStateDistribution(
                            prev_state=prev_state,
                            next_state=next_state,
                            mean_next_state=None,
                            cov_next_state=None
                        )
                        distributions.append(distribution)
                        
                        if len(distributions) >= self._num_samples:
                            break
                    
                    if len(distributions) >= self._num_samples:
                        break
                        
            except Exception as e:
                continue
        
        if len(distributions) < self._num_samples:
            print(f"Warning: Only found {len(distributions)} state pairs, requested {self._num_samples}")
        
        return distributions[:self._num_samples]


#-----------------------------------------------------
# Dataloader.
#-----------------------------------------------------


@gin.configurable
def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 1,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Returns a dataloader of the dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )


def main():
    dists = BMLmoviDataset(
        num_samples=1,
        signal_length=100,
        smnr_db=10,
    )
    states = dists._states
    noisy_observations = dists._noisy_observations[0]
    Cw = dists._Cws[0].squeeze()
    obs_dim = 63

    obs_cov = torch.cov(noisy_observations.reshape(-1, obs_dim).T)
    print(obs_cov.shape)

    # Subtract known noise covariance
    noise_cov = Cw  # Your measurement noise covariance
    H_is_identity = True

    # Estimate state covariance (assuming H = I for your case)
    if H_is_identity:
        obs_changes = noisy_observations[1:] - noisy_observations[:-1]  # Shape: (98, 63)

        # Estimate covariance of observation changes
        change_cov = torch.cov(obs_changes.T)  # Shape: (63, 63)

        # Ensure positive definiteness
        diag_mask = torch.eye(change_cov.shape[0], dtype=torch.bool)
        change_cov[diag_mask] = torch.clamp(change_cov[diag_mask], min=1e-6)
        change_cov = (change_cov + change_cov.T) / 2

    # Use this for baseline prediction
    x_t = states[0, :-1]  # Shape: (N, T-1, 63) - current states
    x_t_plus_1 = states[0, 1:]
    dist_baseline = torch.distributions.MultivariateNormal(x_t, change_cov)
    ll_baseline = dist_baseline.log_prob(x_t_plus_1).mean()
    print('LL baseline', ll_baseline)

    # Get consecutive pairs
    x_t = states[:, :-1]  # Shape: (N, T-1, 63) - current states
    x_t_plus_1 = states[:, 1:]  # Shape: (N, T-1, 63) - next states

    # Flatten to get all transitions
    x_t = x_t.reshape(-1, 63)  # Shape: (N*(T-1), 63)
    x_t_plus_1 = x_t_plus_1.reshape(-1, 63)  # Shape: (N*(T-1), 63)

    # Calculate single temporal std
    temporal_changes = x_t_plus_1 - x_t
    temporal_std = temporal_changes.std().item()  # Single scalar
    print(f"Temporal std: {temporal_std}")

    # Create diagonal covariance
    cov = temporal_std**2 * torch.eye(63)

    # Case 1: Baseline (predict current state)
    mu_baseline = x_t[0]  # Predict no change for first sample
    dist_baseline = torch.distributions.MultivariateNormal(mu_baseline, cov)
    ll_baseline = dist_baseline.log_prob(x_t_plus_1[0])

    # Case 2: Perfect prediction
    mu_perfect = x_t_plus_1[0]  # Perfect prediction for first sample
    dist_perfect = torch.distributions.MultivariateNormal(mu_perfect, cov)
    ll_perfect = dist_perfect.log_prob(x_t_plus_1[0])

    print(f"Baseline log-likelihood (predict current): {ll_baseline.item()}")
    print(f"Perfect prediction log-likelihood: {ll_perfect.item()}")
    print(f"Your model: -20")
    """
    dists = BMLmoviDataset(
        num_samples=1000,
        signal_length=100,
        smnr_db=10,
    )
    
    def validate_tensor(tensor, name):
        if torch.isnan(tensor).any():
            print(f"Warning: {name} contains NaN values")
        if torch.isinf(tensor).any():
            print(f"Warning: {name} contains infinity values")
        if not torch.isfinite(tensor).all():
            print(f"Warning: {name} contains non-finite values")
    states = dists._states
    print("Mean per dimension:")
    print(states.mean(dim=(0,1)))  # Mean across samples and time
    print("\nStd per dimension:")
    print(states.std(dim=(0,1)))   # Std across samples and time
    print(f"\nNumber of dimensions: {states.shape[-1]}")
    print(f"Min std: {states.std(dim=(0,1)).min()}")
    print(f"Max std: {states.std(dim=(0,1)).max()}")
    """

if __name__ == "__main__":
    main()