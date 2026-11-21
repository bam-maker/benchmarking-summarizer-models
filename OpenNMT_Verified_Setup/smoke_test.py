import argparse,sys,time,json,io,re,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'legacy'))
import torch,onmt.opts
from onmt.translate.translator import build_translator
torch.set_num_threads(2)
results=[]
for name in ['lstm','transformer']:
 parser=argparse.ArgumentParser();onmt.opts.translate_opts(parser)
 opt=parser.parse_args(['-model',str(root/(name+'_inference.pt')),'-src',str(root/'article.txt'),'-output',str(root/(name+'_output.txt')),'-beam_size','4','-batch_size','1','-block_ngram_repeat','3','-ignore_when_blocking','.', '</t>','<t>'])
 translator=build_translator(opt,report_score=False,out_file=io.StringIO())
 count=sum(p.numel() for p in translator.model.parameters())
 print(name,'parameters',count,flush=True)
 for level,min_len,max_len in [('short',10,35),('medium',30,65),('detailed',55,100)]:
  translator.min_length=min_len;translator.max_length=max_len
  t=time.perf_counter()
  with torch.inference_mode():
   scores,preds=translator.translate(src_path=str(root/'article.txt'),batch_size=1)
  raw=preds[0][0];clean=re.sub(r'</?t>','',raw);clean=' '.join(clean.split()).replace(' .','.').replace(' ,',',')
  row=dict(model=name,parameters=count,length=level,min_tokens=min_len,max_tokens=max_len,seconds=round(time.perf_counter()-t,3),words=len(clean.split()),summary=clean)
  results.append(row);print(json.dumps(row),flush=True)
(root/'smoke_results.json').write_text(json.dumps(results,indent=2))
