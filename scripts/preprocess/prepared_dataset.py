"""Load a prepared split without normalizing twice.

Map-style dataset compatible with torch.utils.data.DataLoader. Each item is
(float32 NumPy array of shape 3x112x112 in [-1,1], integer first-name label).
PyTorch's default collation converts these NumPy arrays into tensors. Each
worker opens its own read-only HDF5 handle; no torch dependency is required
just to inspect the prepared dataset.
"""
import os
from pathlib import Path
import h5py


class PreparedFaces:
    def __init__(self, path, split='train'):
        if split not in ('train','validation','test'):
            raise ValueError('split must be train, validation, or test')
        self.path=str(Path(path).resolve())
        self.split=split
        self._file=None
        self._pid=None
        with h5py.File(self.path,'r') as f:
            if not bool(f.attrs['complete']):
                raise ValueError('Prepared file is incomplete')
            self.indices=f[f'split_indices/{split}'][:]
            self.class_names=[x.decode() if isinstance(x,bytes) else str(x) for x in f['class_names'][:]]

    def __len__(self):
        return len(self.indices)

    def _handle(self):
        pid=os.getpid()
        if self._file is None or self._pid!=pid:
            if self._file is not None:
                self._file.close()
            self._file=h5py.File(self.path,'r')
            self._pid=pid
        return self._file

    def __getitem__(self,index):
        i=int(self.indices[index]);f=self._handle()
        return f['images'][i],int(f['labels'][i])

    def filename(self,index):
        value=self._handle()['filenames'][int(self.indices[index])]
        return value.decode() if isinstance(value,bytes) else str(value)

    def close(self):
        if self._file is not None:
            self._file.close();self._file=None;self._pid=None

    def __getstate__(self):
        state=self.__dict__.copy();state['_file']=None;state['_pid']=None
        return state
