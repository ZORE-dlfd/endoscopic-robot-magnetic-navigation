"""Read-only desktop progress window. Closing it does not stop training."""
import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def main(folder):
    root = tk.Tk()
    root.title('磁场与梯度网络 · GPU 训练进度')
    root.geometry('960x650')
    root.minsize(800, 530)
    root.configure(bg='#eff3f8')
    title = tk.Label(root, text='磁场与梯度网络训练', font=('Microsoft YaHei', 20, 'bold'), bg='#eff3f8')
    title.pack(anchor='w', padx=24, pady=(18, 4))
    detail = tk.StringVar(value='正在准备数据和 GPU…')
    tk.Label(root, textvariable=detail, font=('Microsoft YaHei', 11), bg='#eff3f8', justify='left').pack(anchor='w', padx=24)
    progress = ttk.Progressbar(root, maximum=100)
    progress.pack(fill='x', padx=24, pady=12)
    summary = tk.StringVar()
    tk.Label(root, textvariable=summary, font=('Microsoft YaHei', 12), bg='#eff3f8', justify='left').pack(anchor='w', padx=24)
    canvas = tk.Canvas(root, bg='white', highlightthickness=0, height=300)
    canvas.pack(fill='both', expand=True, padx=24, pady=14)
    tk.Label(root, text='曲线：训练损失（对数刻度）；外部验证尚未接入。关闭窗口不会停止训练。',
        font=('Microsoft YaHei', 10), bg='#eff3f8').pack(anchor='w', padx=24)
    tk.Label(root, text=str(folder), font=('Microsoft YaHei', 9), bg='#eff3f8', wraplength=900).pack(anchor='w', padx=24, pady=(4, 16))

    def refresh():
        config = read_json(folder/'config.json')
        status = read_json(folder/'status.json')
        audit = read_json(folder/'data_audit.json').get('training', {})
        rows = []
        try:
            for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        except OSError:
            pass
        epochs = config.get('epochs', 500)
        epoch = rows[-1]['epoch'] if rows else 0
        elapsed = rows[-1]['elapsed_s'] if rows else 0.
        finished = status.get('status') == 'training_finished'
        failed = status.get('status') == 'failed'
        label = '训练结束，等待独立验证' if finished else ('训练失败：'+status.get('error', '') if failed else 'GPU 训练中' if epoch else '数据审计 / GPU 初始化')
        detail.set(f'{label}\n设备：{status.get("gpu", "NVIDIA GeForce RTX 2080 Ti / CUDA")}    病例：{len(audit.get("case_ids", []))}    样本：{audit.get("unique_samples", "准备中")}')
        progress['value'] = 100*epoch/max(epochs, 1)
        eta = '前几轮完成后估算'
        if len(rows) >= 3:
            past = rows[max(0, len(rows)-21)]
            seconds = (rows[-1]['elapsed_s']-past['elapsed_s'])/max(epoch-past['epoch'], 1)
            remain = max(0, epochs-epoch)*seconds
            eta = f'约 {remain/60:.1f} 分钟；预计 {datetime.now()+timedelta(seconds=remain):%H:%M:%S} 完成'
        loss = f'{rows[-1]["training_loss"]:.6g}' if rows else '—'
        summary.set(f'轮数：{epoch} / {epochs}    损失：{loss}    已用：{elapsed/60:.1f} 分钟\n'+('已保存 last.pt；模型精度尚待外部验证' if finished else '预计剩余：'+eta))
        canvas.delete('all')
        w, h = max(canvas.winfo_width(), 300), max(canvas.winfo_height(), 200)
        left, top, right, bottom = 70, 25, w-25, h-40
        canvas.create_line(left, top, left, bottom, right, bottom, fill='#9ba8b8')
        canvas.create_text((left+right)/2, h-13, text='训练轮数', fill='#52647b')
        values = [math.log10(max(row['training_loss'], 1e-15)) for row in rows]
        if values:
            lo, hi = min(values), max(values)
            if hi-lo < .01:
                hi = lo+.01
            points = []
            for row, value in zip(rows, values):
                points.extend([left+(right-left)*row['epoch']/epochs, bottom-(bottom-top)*(value-lo)/(hi-lo)])
            for value in (lo, (lo+hi)/2, hi):
                y = bottom-(bottom-top)*(value-lo)/(hi-lo)
                canvas.create_line(left, y, right, y, fill='#e9eef5')
                canvas.create_text(left-8, y, text=f'{10**value:.3g}', anchor='e', fill='#52647b')
            if len(points) >= 4:
                canvas.create_line(*points, fill='#2474d6', width=2)
        root.after(2000, refresh)

    root.after(100, lambda: (root.lift(), root.attributes('-topmost', True)))
    root.after(1800, lambda: root.attributes('-topmost', False))
    refresh()
    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    main(Path(parser.parse_args().folder))
