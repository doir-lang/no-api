#!/usr/bin/env python3
"""Checks noapi's header documentation against AGENTS.md: every public definition has a Doxygen comment, every function's comment has an example, and every example compiles and passes on the backend being built.

Each `@code` block becomes a doctest test case run against a headless queue named `queue`, created from the backend objects named `backend` (a GpuVulkanDefault or GpuWebGPUDefault). A plain `@code` block is compiled as both C and C++; `@code{.cpp}` marks one as C++ only. `#line` directives point failures back at the example's line in the header. `assert` in an example reports a failure rather than aborting.

Usage:
  tools/doctest.py [BUILD_DIR...]    build and run the examples in each build directory (default: build), then report documentation problems
  tools/doctest.py --check           only report documentation problems
  tools/doctest.py --generate DIR    write the example sources into DIR (run by the build)
"""
import os, re, subprocess, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
headers = ['allocator.h', 'compat.h', 'compute.h', 'graphics.h', 'samplers.h', 'shaders.h', 'surface.h', 'sync.h',
	'vulkan/noapi.h', 'vulkan/noapi.hpp', 'webgpu/noapi.h', 'webgpu/noapi.hpp']
# Each backend's examples are compiled only against that backend; the shared headers' examples against either.
backend_of = {'vulkan/noapi.h': 'vulkan', 'vulkan/noapi.hpp': 'vulkan', 'webgpu/noapi.h': 'webgpu', 'webgpu/noapi.hpp': 'webgpu'}


def doc_blocks(lines):
	"""Yields (end, body) for each `/** */` comment: the index of the line after it, and its (line number, text) pairs."""
	i = 0
	while i < len(lines):
		s = lines[i].strip()
		if not (s.startswith('/**') and not s.startswith('/**/')):
			i += 1
			continue
		first = s[3:]
		if first.endswith('*/'):
			yield i + 1, [(i + 1, first[:-2].strip())]
			i += 1
			continue
		body = [(i + 1, first.strip())] if first.strip() else []
		i += 1
		while i < len(lines) and '*/' not in lines[i]:
			body.append((i + 1, re.sub(r'^\s*\*\s?', '', lines[i])))
			i += 1
		yield i + 1, body
		i += 1


def declaration(lines, j):
	"""The declaration starting at line j, joined up to its first `;` or `{`, with any template header kept."""
	text = []
	while j < len(lines):
		s = lines[j].split('//')[0].strip()
		text.append(s)
		if s.endswith((';', '{', '}')) or s.startswith('#'):
			break
		j += 1
	return ' '.join(text)


def is_function(decl):
	if decl.startswith(('#', 'typedef', 'struct', 'enum', 'union', 'using', 'namespace', 'class')):
		return False
	head = re.sub(r'^template\s*<.*?>\s*', '', decl)
	if '=' in head.split('(')[0]:
		return False
	return bool(re.match(r'[\w:<>,*&\s]+?\bgpu\w*\s*\(', head))


def name_of(decl):
	found = re.findall(r'\b(gpu\w+|Gpu\w+|GPU_\w+|NOAPI_\w+|[A-Z_]{3,})\b', decl)
	return found[0] if found else decl[:40]


def check(path, lines, problems, examples):
	documented = set()
	for end, body in doc_blocks(lines):
		j = end
		while j < len(lines) and not lines[j].strip():
			j += 1
		documented.add(j)
		decl = declaration(lines, j) if j < len(lines) else ''
		found = []
		code = None
		for line, text in body:
			marker = text.strip()
			if code is None and marker.startswith('@code'):
				code = {'line': line + 1, 'code': [], 'cpp_only': marker == '@code{.cpp}', 'name': name_of(decl)}
			elif code is not None and marker == '@endcode':
				found.append(code)
				code = None
			elif code is not None:
				code['code'].append(text)
		if is_function(decl) and not found:
			problems.append(f'{path}:{j + 1}: documented without an example: {decl}')
		for example in found:
			example['path'] = path
		examples += found

	# A declaration at file scope with nothing documenting it. One comment covers a run of declarations with no blank line between them, such as the C and C++ spellings of a constant either side of an #ifdef, and so does a run that starts with the other spelling of something documented.
	skip = ('//', '/*', '*', '#pragma', '#include', '#if', '#el', '#endif', '#undef', '#error', 'NOAPI_EXTERN_C', '}', '{',
		'extern "C"', 'static_assert', 'namespace')
	# The definitions behind the opaque handles live in noapi.hpp; compat.h counts them as backend internals.
	internal = re.compile(r'(struct Gpu\w+ \{|.*\b\w+::\w+\()') if path.endswith('.hpp') else re.compile(r'.*\b\w+::\w+\(')
	# The other spelling of something documented under the opposite branch of an #ifdef __cplusplus is covered by that comment.
	names = {name_of(declaration(lines, j)) for j in documented if j < len(lines)}
	depth = 0
	statement = False
	for k, line in enumerate(lines):
		s = line.split('//')[0].strip()
		if depth == 0 and not statement and s and not s.startswith(skip) and not internal.match(s) and k not in documented and name_of(s) not in names:
			above = k - 1
			while above >= 0 and lines[above].strip() and not lines[above].strip().endswith('*/') and above not in documented and name_of(lines[above].split('//')[0].strip()) not in names:
				above -= 1
			if not (above >= 0 and (above in documented or lines[above].strip().endswith('*/') or name_of(lines[above].split('//')[0].strip()) in names)):
				problems.append(f'{path}:{k + 1}: undocumented: {s}')
		if depth == 0 and s and not s.startswith(('#', '//', '/*', '*')):
			statement = not s.endswith((';', '{', '}', ')')) or s.startswith('template')
		if s.startswith('#') and s.endswith('\\'):
			statement = True
		elif s.startswith('#'):
			statement = False
		depth += s.count('{') - s.count('}')
		depth = max(depth, 0)


def collect():
	problems = []
	examples = []
	for path in headers:
		check(path, open(os.path.join(root, path), encoding='utf-8').read().split('\n'), problems, examples)
	return problems, examples


def generate(out):
	"""Writes one C++ and one C translation unit per backend; the build picks the pair for the backend it links."""
	_, examples = collect()
	os.makedirs(out, exist_ok=True)
	for backend in ('vulkan', 'webgpu'):
		chosen = [e for e in examples if backend_of.get(e['path'], backend) == backend]
		cpp = ['// Generated by tools/doctest.py from the headers\' @code examples; edit those instead.', '#include <doctest/doctest.h>',
			'#include "harness.h"', '']
		c = ['/* Generated by tools/doctest.py from the headers\' @code examples; edit those instead. */', '#include "harness.h"', '',
			'int noapi_doctest_c_unit;']
		for k, example in enumerate(chosen):
			source = os.path.join(root, example['path']).replace('\\', '/')
			where = f'{example["path"]}:{example["line"]} {example["name"]}'
			body = [f'#line {example["line"]} "{source}"', *example['code'], f'#line {10000 + k} "doctest"']
			cpp += [f'#line {example["line"]} "{source}"', f'TEST_CASE("{where}") {{', '\tGpuQueue* queue = noapi_doctest_queue();', '\tGpuDoctestBackend backend = noapi_doctest_backend();', '\t(void)backend;', '\tnoapi_doctest_begin();', '\t{',
				*body, '\t}', '\tnoapi_doctest_end();', '}', '']
			if not example['cpp_only']:
				c += [f'void noapi_doctest_c_{k}(GpuQueue* queue);', f'void noapi_doctest_c_{k}(GpuQueue* queue) {{', '\tGpuDoctestBackend backend = noapi_doctest_backend();', '\t(void)queue;', '\t(void)backend;',
					*body, '}', '']
				cpp += [f'extern "C" void noapi_doctest_c_{k}(GpuQueue* queue);', f'#line {example["line"]} "{source}"', f'TEST_CASE("{where} (C)") {{', '\tnoapi_doctest_begin();',
					f'\tnoapi_doctest_c_{k}(noapi_doctest_queue());', '\tnoapi_doctest_end();', '}', '']
		write(os.path.join(out, f'examples_{backend}.cpp'), '\n'.join(cpp) + '\n')
		write(os.path.join(out, f'examples_{backend}.c'), '\n'.join(c) + '\n')


def write(path, text):
	"""Leaves an unchanged file alone, so regenerating does not rebuild the examples."""
	if os.path.exists(path) and open(path, encoding='utf-8').read() == text:
		return
	open(path, 'w', encoding='utf-8').write(text)


def run(build_dirs):
	failed = False
	for build in build_dirs:
		build = os.path.join(root, build)
		if subprocess.run(['cmake', '--build', build, '--target', 'noapi-doctests'], cwd=root).returncode != 0:
			failed = True
			continue
		exe = os.path.join(build, 'doctests', 'noapi-doctests' + ('.exe' if os.name == 'nt' else ''))
		if subprocess.run([exe], cwd=root).returncode != 0:
			failed = True
	return failed


def main():
	args = sys.argv[1:]
	if args[:1] == ['--generate']:
		generate(args[1])
		return 0
	failed = False if args[:1] == ['--check'] else run(args or ['build'])
	problems, _ = collect()
	for problem in problems:
		print(problem)
	if problems:
		print(f'doctest: {len(problems)} documentation problems')
	return 1 if failed or problems else 0


if __name__ == '__main__':
	sys.exit(main())
