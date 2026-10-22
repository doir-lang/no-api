#!/usr/bin/env python3
"""Measures how much of each backend the header examples run, using gcov. Each backend gets its own instrumented build, build-cov-<backend>, which runs the doc examples headlessly.

Usage: tools/coverage.py [-v] [vulkan] [webgpu]    (-v lists the uncovered lines; with no backend named, both are measured)
Set NOAPI_SLANG_ROOT to build against a prebuilt Slang rather than the submodule.
"""
import glob, json, os, subprocess, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gcov = os.environ.get('GCOV', 'gcov')


def build_and_run(backend):
	build = os.path.join(root, f'build-cov-{backend}')
	configure = ['cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DNOAPI_ENABLE_COVERAGE=ON',
		'-DNOAPI_ENABLE_LTO=OFF', '-DNOAPI_BUILD_DOCTESTS=ON', f'-DNOAPI_DEFAULT_BACKEND={backend}',
		f'-DNOAPI_ENABLE_VULKAN={"ON" if backend == "vulkan" else "OFF"}', f'-DNOAPI_ENABLE_WEBGPU={"ON" if backend == "webgpu" else "OFF"}']
	if os.environ.get('NOAPI_SLANG_ROOT'):
		configure.append('-DNOAPI_SLANG_ROOT=' + os.environ['NOAPI_SLANG_ROOT'])
	if subprocess.run(configure, cwd=root).returncode != 0 or subprocess.run(['cmake', '--build', build, '--target', 'noapi-doctests'], cwd=root).returncode != 0:
		sys.exit(1)

	# Counters accumulate across runs, so a stale .gcda would credit lines this run never reached.
	objects = os.path.join(build, backend, 'CMakeFiles', f'noapi-{backend}.dir')
	for old in glob.glob(os.path.join(objects, '**', '*.gcda'), recursive=True):
		os.remove(old)
	run = subprocess.run([os.path.join(build, 'doctests', 'noapi-doctests')], cwd=root, capture_output=True, text=True)
	with open(os.path.join(build, 'doctests', 'run.log'), 'w') as log:
		log.write(run.stdout + run.stderr)
	if run.returncode != 0:
		print(run.stdout + run.stderr)
		print(f'coverage: the {backend} examples failed; the numbers below only count what ran before that')
	return objects


def read_counts(objects):
	"""Maps each of noapi's own sources to {line: (hits, text)}; lines gcov reports nothing for are not code."""
	files = {}
	for data in glob.glob(os.path.join(objects, '**', '*.gcda'), recursive=True):
		result = subprocess.run([gcov, '--json-format', '--stdout', data], cwd=os.path.dirname(data), capture_output=True, text=True)
		if result.returncode != 0:
			sys.exit(result.stderr)
		for document in result.stdout.splitlines():
			for entry in json.loads(document)['files']:
				path = os.path.normpath(os.path.join(os.path.dirname(data), entry['file']))
				relative = os.path.relpath(path, root)
				if relative.startswith(('..', 'build', 'external')):
					continue
				counts = files.setdefault(relative, {})
				for line in entry['lines']:
					# An inline function is counted once per object it was compiled into; any one running a line covers it.
					before = counts.get(line['line_number'], 0)
					counts[line['line_number']] = max(before, line['count'])
	return files


def report(files, verbose):
	all_covered = all_total = 0
	for name in sorted(files):
		counts = files[name]
		missed = sorted(n for n, hits in counts.items() if hits == 0)
		total = len(counts)
		covered = total - len(missed)
		flag = f'   <-- {len(missed)} uncovered' if missed else ''
		print(f'{name:<40} {100 * covered / total:6.2f}%  ({covered}/{total}){flag}' if total else f'{name:<40} {"n/a":>6}')
		if verbose and missed:
			text = open(os.path.join(root, name), encoding='utf-8', errors='replace').read().split('\n')
			for n in missed:
				print(f'      {n:5}: {text[n - 1].strip() if n <= len(text) else ""}')
		all_covered += covered
		all_total += total
	print('-' * 70)
	percent = 100 * all_covered / all_total if all_total else 100
	print(f'{"TOTAL":<40} {percent:6.2f}%  ({all_covered}/{all_total})')


if __name__ == '__main__':
	args = sys.argv[1:]
	backends = [a for a in args if a in ('vulkan', 'webgpu')] or ['vulkan', 'webgpu']
	for backend in backends:
		files = read_counts(build_and_run(backend))
		print(f'\n{backend}:')
		report(files, '-v' in args)
