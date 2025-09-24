import torch

from data_loader.dataset import LorenzAttractor, Lorenz96Attractor, BMLmoviDataset
from data_loader.dataset import LorenzAttractorTest, Lorenz96AttractorTest, BMLmoviTestDataset

from base import BaseDataLoader


_NUM_SAMPLES = 1250


class LorenzDataLoader(BaseDataLoader):
    def __init__(self, 
                 num_samples=1250,
                 signal_length=100, 
                 smnr_db=10,
                 batch_size=20,
                 shuffle=True,
                 validation_split=0.2,
                 num_workers=1):
        
        self.dataset = LorenzAttractor(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db
        )
        
        super().__init__(
            self.dataset, 
            batch_size, 
            shuffle, 
            validation_split, 
            num_workers,
            collate_fn=self._lorenz_collate_fn
        )
    
    def _lorenz_collate_fn(self, batch):
        """Custom collate function for Lorenz data including Cw"""
        states, observations, noisy_observations, Cws = zip(*batch)
        
        states = torch.stack(states)
        observations = torch.stack(observations) 
        noisy_observations = torch.stack(noisy_observations)
        Cws = torch.stack(Cws)
        
        batch_size, seq_length, dim = noisy_observations.shape
        
        # Use noisy observations as input
        x = noisy_observations
        x_reversed = torch.flip(x, dims=[1])
        x_mask = torch.ones(batch_size, seq_length)
        x_seq_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
        
        # Return Cw as additional information
        return x, x_reversed, x_mask, x_seq_lengths, Cws


class Lorenz96DataLoader(BaseDataLoader):
    def __init__(self, 
                 dimension=8,
                 num_samples=1250,
                 signal_length=100, 
                 smnr_db=10,
                 K=5,
                 batch_size=20,
                 shuffle=True,
                 validation_split=0.2,
                 num_workers=1):
        
        self.dataset = Lorenz96Attractor(
            dimension=dimension,
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            K=K
        )
        
        super().__init__(
            self.dataset, 
            batch_size, 
            shuffle, 
            validation_split, 
            num_workers,
            collate_fn=self._lorenz96_collate_fn
        )
    
    def _lorenz96_collate_fn(self, batch):
        """Custom collate function for Lorenz96 data"""
        states, observations, noisy_observations, Cws = zip(*batch)
        
        states = torch.stack(states)
        observations = torch.stack(observations) 
        noisy_observations = torch.stack(noisy_observations)
        Cws = torch.stack(Cws)
        
        batch_size, seq_length, dim = noisy_observations.shape
        
        x = noisy_observations
        x_reversed = torch.flip(x, dims=[1])
        x_mask = torch.ones(batch_size, seq_length)
        x_seq_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
        
        return x, x_reversed, x_mask, x_seq_lengths, Cws


class BMLmoviDataLoader(BaseDataLoader):
    def __init__(self, 
                 num_samples=1250,
                 signal_length=100, 
                 smnr_db=10,
                 base_path="/mimer/NOBACKUP/groups/naiss2025-22-438/BMLmovi",
                 batch_size=20,
                 shuffle=True,
                 validation_split=0.2,
                 num_workers=1):
        
        self.dataset = BMLmoviDataset(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            base_path=base_path
        )
        
        super().__init__(
            self.dataset, 
            batch_size, 
            shuffle, 
            validation_split, 
            num_workers,
            collate_fn=self._bmlmovi_collate_fn
        )
    
    def _bmlmovi_collate_fn(self, batch):
        """Custom collate function for BMLmovi data"""
        states, observations, noisy_observations, Cws = zip(*batch)
        
        states = torch.stack(states)
        observations = torch.stack(observations) 
        noisy_observations = torch.stack(noisy_observations)
        Cws = torch.stack(Cws)
        
        batch_size, seq_length, dim = noisy_observations.shape
        
        x = noisy_observations
        x_reversed = torch.flip(x, dims=[1])
        x_mask = torch.ones(batch_size, seq_length)
        x_seq_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
        
        return x, x_reversed, x_mask, x_seq_lengths, Cws


class LorenzTestDataLoader(BaseDataLoader):
    def __init__(self, 
                 num_samples=1000,
                 signal_length=100,
                 batch_size=20,
                 shuffle=False,
                 num_workers=1):
        
        self.dataset = LorenzAttractorTest(
            num_samples=num_samples,
            signal_length=signal_length
        )
        
        super().__init__(
            self.dataset, 
            batch_size, 
            shuffle, 
            0.0,  # No validation split for test data
            num_workers,
            collate_fn=self.dataset.collate_fn
        )

class Lorenz96TestDataLoader(BaseDataLoader):
    def __init__(self, 
                 dimension=8,
                 num_samples=1000,
                 signal_length=100,
                 K=1,
                 batch_size=20,
                 shuffle=False,
                 num_workers=1):
        
        self.dataset = Lorenz96AttractorTest(
            dimension=dimension,
            num_samples=num_samples,
            signal_length=signal_length,
            K=K
        )
        
        super().__init__(
            self.dataset, 
            batch_size, 
            shuffle, 
            0.0,
            num_workers,
            collate_fn=self.dataset.collate_fn
        )

class BMLmoviTestDataLoader(BaseDataLoader):
    def __init__(self, 
                 num_samples=1000,
                 base_path="/mimer/NOBACKUP/groups/naiss2025-22-438/BMLmovi",
                 batch_size=20,
                 shuffle=False,
                 num_workers=1):
        
        self.dataset = BMLmoviTestDataset(
            num_samples=num_samples,
            base_path=base_path
        )
        
        super().__init__(
            self.dataset, 
            batch_size, 
            shuffle, 
            0.0,
            num_workers,
            collate_fn=self.dataset.collate_fn
        )