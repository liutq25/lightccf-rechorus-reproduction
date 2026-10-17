"""Compile the standalone Chinese report with XeLaTeX until references settle."""
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / '复现报告.tex')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if source.suffix.lower() != '.tex':
        raise ValueError('需要 .tex 源文件')
    engine = shutil.which('xelatex')
    if engine is None:
        raise RuntimeError('未找到 XeLaTeX；请安装含ctex、pgfplots、Fandol字体的TeX发行版。')
    build = source.parent / 'tmp/latex_build'
    build.mkdir(parents=True, exist_ok=True)
    command = [engine, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               '-no-shell-escape', '-output-directory=' + str(build), source.name]
    for pass_number in range(1, 5):
        log_path = build / f'compiler-pass-{pass_number}.txt'
        with log_path.open('wb') as stream:
            result = subprocess.run(command, cwd=source.parent,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'第{pass_number}次编译失败，诊断保存在{log_path}')
        final_log = (build / source.with_suffix('.log').name).read_text(encoding='utf-8', errors='replace')
        needs_rerun = ('Rerun to get cross-references right' in final_log
                       or 'Rerun to get outlines right' in final_log
                       or 'undefined references' in final_log)
        if pass_number >= 2 and not needs_rerun:
            break
    else:
        raise RuntimeError('四次编译后引用仍未稳定，请检查编译日志。')
    generated = build / source.with_suffix('.pdf').name
    destination = source.with_suffix('.pdf')
    shutil.copyfile(generated, destination)
    for issue in ('Missing character:', 'undefined references'):
        if issue in final_log:
            raise RuntimeError(f'PDF生成，但存在需要修复的诊断：{issue}')
    print(f'XeLaTeX编译成功（{pass_number}次）：{destination}')


if __name__ == '__main__':
    main()
