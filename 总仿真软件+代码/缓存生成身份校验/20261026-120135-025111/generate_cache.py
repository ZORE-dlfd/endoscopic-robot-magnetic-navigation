"""Generate all predicted depth frames from the exact delivered camera replay."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
import bootstrap


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=bootstrap.ASSETS)
    parser.add_argument('--cache-root', type=Path, default=bootstrap.ROOT/'软件（深度缓存直接播放）'/'缓存数据')
    parser.add_argument('--status', type=Path, default=bootstrap.ROOT/'构建'/'multimodal-20261026'/'cache_status.json')
    parser.add_argument('--model-repo', type=Path, default=bootstrap.MODEL_REPO)
    parser.add_argument('--checkpoint', type=Path, default=bootstrap.WEIGHT)
    parser.add_argument('--limit', type=int, default=0, help='optional debug frame count; partial caches are never finalized')
    args = parser.parse_args()
    bootstrap.ASSETS = args.assets.resolve()
    bootstrap.WEIGHT = args.checkpoint.resolve()
    import sys
    sys.path.insert(0, str(args.model_repo.resolve()))
    from magnetic_app.depth.cache import DepthMapCache, route_signature
    from rendering import Renderer
    from depth_model import DepthModel
    import numpy as np
    started = time.perf_counter()
    renderer = cache = None
    def status(stage, **extra):
        payload = dict(stage=stage, pid=os.getpid(), wall_seconds=round(time.perf_counter()-started, 2), **extra)
        write_json(args.status, payload)
        print(json.dumps(payload, ensure_ascii=True), flush=True)
    try:
        status('initializing_renderer')
        renderer = Renderer()
        signature = route_signature(bootstrap.ASSETS)
        count = len(renderer.replay.camera_times)
        directory = args.cache_root.resolve()/signature
        cache = DepthMapCache(directory, signature, count)
        if cache.is_valid:
            status('completed', signature=signature, frame_count=count, completed_count=count, cache_directory=str(directory))
            return
        status('loading_network', signature=signature, frame_count=count)
        model = DepthModel()
        first = cache.begin((518,518))
        last = min(count, first+args.limit) if args.limit else count
        processing_started = time.perf_counter()
        for index in range(first, last):
            t = float(renderer.replay.camera_times[index])
            rgb = renderer.rgb(renderer.frame(t))
            depth, latency = model.predict(rgb)
            if not np.isfinite(depth).all() or float(np.abs(depth).max()) > np.finfo(np.float16).max:
                raise ValueError(f'invalid depth frame {index}')
            cache.write(index, depth)
            if index%25 == 0 or index+1 == last:
                elapsed = time.perf_counter()-processing_started
                rate = (index-first+1)/max(elapsed,1e-6)
                status('generating', signature=signature, frame_count=count, completed_count=index+1,
                       progress_pct=round(100*(index+1)/count,2), simulation_time_s=t,
                       eta_seconds=round((count-index-1)/rate,1), inference_s=latency,
                       cache_directory=str(directory))
        if last == count:
            cache.finalize()
            write_json(directory/'generation.json', dict(signature=signature, frame_count=count,
                       source_assets=str(bootstrap.ASSETS), model=model.metadata,
                       camera_source='delivered replay.camera_times and frame(t), insertion fusion + original withdrawal PD',
                       code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       wall_seconds=time.perf_counter()-started))
            status('completed', signature=signature, frame_count=count, completed_count=count,
                   cache_directory=str(directory), progress_pct=100.)
        else:
            status('partial_debug', signature=signature, completed_count=last, frame_count=count)
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        if cache is not None:
            cache.close_partial()
        if renderer is not None:
            renderer.close()


if __name__ == '__main__':
    main()
