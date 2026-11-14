"""Read status and download outputs with existing Kaggle auth, no secret logs."""
import argparse,json,time,threading,queue
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    p=argparse.ArgumentParser();p.add_argument('--kernel',default='mohrezsayeh/geometry-pi-deeponet-main');p.add_argument('--output',default='runs/kaggle');p.add_argument('--download',action='store_true');p.add_argument('--logs',action='store_true');p.add_argument('--seconds',type=int,default=30);a=p.parse_args()
    api=KaggleApi();api.authenticate();status=api.kernels_status(a.kernel);print(status)
    if a.logs:
        q=queue.Queue()
        def stream():
            try:
                for event in api.kernels_logs_stream(a.kernel):q.put(event)
            except Exception as e:q.put({'error_type':type(e).__name__})
        threading.Thread(target=stream,daemon=True).start();end=time.monotonic()+a.seconds
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        with open(out/'live-log.jsonl','w') as f:
            while time.monotonic()<end:
                try:
                    event=q.get(timeout=1);f.write(json.dumps(event)+'\n');f.flush()
                    try:record=json.loads(event.get('data',''))
                    except ValueError:continue
                    if 'step' in record:print(json.dumps(record),flush=True)
                except queue.Empty:pass
    if a.download:
        try:api.kernels_output(a.kernel,path=a.output,quiet=True)
        except Exception as e:
            # Do not print exception strings: some clients include signed URLs.
            print('Download failed:',type(e).__name__,'Check allowed domains: www.kaggleusercontent.com, storage.googleapis.com')
            raise SystemExit(1) from None
if __name__=='__main__':main()
