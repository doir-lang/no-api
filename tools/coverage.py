#!/usr/bin/env python3
"""Measures how much of each backend the header examples and the example programs run, using gcov. Each backend gets its own instrumented build, build-cov-<backend>, which runs the doc examples headlessly and then each program in examples/ for a fixed number of frames. Unless VK_DRIVER_FILES picks a driver, everything runs twice, on Mesa's software Vulkan driver (llvmpipe, when installed) and on the default one, and the counts of the two are combined.

Usage: tools/coverage.py [-v] [--doctests-only] [vulkan] [webgpu]
  -v lists the uncovered lines; --doctests-only skips the example programs, which need a display; with no backend named, both are measured.
Set NOAPI_SLANG_ROOT to build against a prebuilt Slang rather than the submodule.
"""
import glob, json, os, subprocess, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gcov = os.environ.get('GCOV', 'gcov')
example_frames = 120
example_timeout = 300


def vulkan_drivers():
	"""The passes to run, as (name, environment) pairs. Mesa's llvmpipe implements the optional features (mesh shaders) a GPU may lack, while the GPU takes the device dependent paths llvmpipe never does (a texture heap that has to be snapshotted); the counters of both add up."""
	if os.environ.get('VK_DRIVER_FILES') or os.environ.get('VK_ICD_FILENAMES'):
		return [('the chosen driver', dict(os.environ))]
	passes = [('the default driver', dict(os.environ))]
	for directory in ('/usr/share/vulkan/icd.d', '/usr/local/share/vulkan/icd.d', '/etc/vulkan/icd.d'):
		found = sorted(glob.glob(os.path.join(directory, 'lvp_icd*.json')))
		if found:
			return [('llvmpipe', dict(os.environ, VK_DRIVER_FILES=found[0]))] + passes
	print('coverage: llvmpipe (lvp_icd.json) not found; measuring on the default Vulkan driver alone, so paths it lacks features for stay uncovered')
	return passes


def example_programs():
	return sorted(os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(root, 'examples', '[0-9]*_*.cpp')))


def run_logged(command, log_path, env, timeout=None):
	"""Runs command with its output written to log_path; prints the output and returns False if it failed."""
	try:
		run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, env=env)
		output, ok = run.stdout + run.stderr, run.returncode == 0
	except subprocess.TimeoutExpired:
		output, ok = f'timed out after {timeout}s\n', False
	with open(log_path, 'w') as log:
		log.write(output)
	if not ok:
		print(output)
	return ok


def build_and_run(backend, examples):
	build = os.path.join(root, f'build-cov-{backend}')
	configure = ['cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DNOAPI_ENABLE_COVERAGE=ON',
		'-DNOAPI_ENABLE_LTO=OFF', '-DNOAPI_BUILD_DOCTESTS=ON', f'-DNOAPI_DEFAULT_BACKEND={backend}',
		f'-DNOAPI_ENABLE_VULKAN={"ON" if backend == "vulkan" else "OFF"}', f'-DNOAPI_ENABLE_WEBGPU={"ON" if backend == "webgpu" else "OFF"}']
	if os.environ.get('NOAPI_SLANG_ROOT'):
		configure.append('-DNOAPI_SLANG_ROOT=' + os.environ['NOAPI_SLANG_ROOT'])
	targets = ['noapi-doctests'] + (example_programs() if examples else [])
	if subprocess.run(configure, cwd=root).returncode != 0 or subprocess.run(['cmake', '--build', build, '--target', *targets], cwd=root).returncode != 0:
		sys.exit(1)

	# Counters accumulate across runs, so a stale .gcda would credit lines this run never reached.
	objects = os.path.join(build, backend, 'CMakeFiles', f'noapi-{backend}.dir')
	for old in glob.glob(os.path.join(objects, '**', '*.gcda'), recursive=True):
		os.remove(old)
	logs = os.path.join(build, 'coverage-logs')
	os.makedirs(logs, exist_ok=True)
	for k, (driver, env) in enumerate(vulkan_drivers()):
		print(f'coverage: running the {backend} build on {driver}')
		log = os.path.join(logs, f'{k}-doctests.log')
		if not run_logged([os.path.join(build, 'doctests', 'noapi-doctests')], log, env):
			print(f'coverage: the {backend} doc examples failed on {driver} (log in {os.path.relpath(log, root)}); a run that crashed contributes nothing below')
		if not examples:
			continue
		for name in example_programs():
			log = os.path.join(logs, f'{k}-{name}.log')
			if not run_logged([os.path.join(build, 'examples', name)], log, dict(env, NOAPI_EXAMPLE_FRAMES=str(example_frames)), example_timeout):
				print(f'coverage: example {name} failed on {driver} (log in {os.path.relpath(log, root)}); a run that crashed contributes nothing below')
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
	examples = '--doctests-only' not in args
	if examples and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
		print('coverage: no display to open the example programs on; measuring the doc examples only')
		examples = False
	for backend in backends:
		files = read_counts(build_and_run(backend, examples))
		print(f'\n{backend}:')
		report(files, '-v' in args)
