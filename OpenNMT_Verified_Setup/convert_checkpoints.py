import sys,torch,pickle,types
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'legacy'))
import onmt
class TrainingState:
 def __setstate__(self,state): self.__dict__.update(state)
class InferenceUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if module.startswith('torch.optim') or module=='onmt.Optim':return TrainingState
  return super().find_class(module,name)
compat_pickle=types.ModuleType('compat_pickle')
compat_pickle.Unpickler=InferenceUnpickler
compat_pickle.load=pickle.load
compat_pickle.__version__=pickle.format_version
for name in ['lstm','transformer']:
 p=str(root/(name+'.pt'))
 c=torch.load(p,map_location='cpu',weights_only=False,pickle_module=compat_pickle)
 print(name, 'checkpoint loaded')
 print('state_tensor_elements',sum(t.numel() for t in c['model'].values()),'generator',sum(t.numel() for t in c['generator'].values()))
 print('vocab',[(k,len(v)) for k,v in c['vocab']]);print('keys',list(c['model'])[:20])
 c.pop('optim',None)
 torch.save(c,str(root/(name+'_inference.pt')))
