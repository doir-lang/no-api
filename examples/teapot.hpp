#pragma once

#include "common.hpp"
#include "tiny_obj_loader.h"

namespace test {

// The teapot the examples draw, as a position and a smooth normal per vertex
struct MeshVertex {
	std::array<float, 3> position;
	std::array<float, 3> normal;
};

// Loads the Utah teapot, centred and scaled to fit a 2 unit cube, with normals averaged from the faces around each vertex.
inline std::vector<MeshVertex> load_teapot(std::vector<uint32_t>& out_indices) {
	constexpr auto path = EXAMPLE_ASSET_DIR "/teapot.obj";

	tinyobj::ObjReaderConfig config;
	config.triangulate = true;
	config.mtl_search_path = EXAMPLE_ASSET_DIR;

	tinyobj::ObjReader reader;
	if(!reader.ParseFromFile(path, config))
		throw std::runtime_error(std::string("Failed to load ") + path + ": " + reader.Error());
	if(!reader.Warning().empty()) std::println("[obj] {}", reader.Warning());

	auto& attrib = reader.GetAttrib();
	auto count = attrib.vertices.size() / 3;

	std::vector<MeshVertex> vertices(count);
	for(size_t i = 0; i < count; ++i)
		vertices[i] = {{attrib.vertices[i * 3], attrib.vertices[i * 3 + 1], attrib.vertices[i * 3 + 2]}, {0, 0, 0}};

	out_indices.clear();
	for(auto& shape: reader.GetShapes())
		for(size_t face = 0; face + 2 < shape.mesh.indices.size(); face += 3) {
			std::array<uint32_t, 3> corner;
			for(int i = 0; i < 3; ++i)
				corner[i] = uint32_t(shape.mesh.indices[face + i].vertex_index);

			auto at = [&](uint32_t index) {
				auto& v = vertices[index].position;
				return vec3{v[0], v[1], v[2]};
			};
			auto weighted = cross(at(corner[1]) - at(corner[0]), at(corner[2]) - at(corner[0]));

			for(int i = 0; i < 3; ++i) {
				auto& normal = vertices[corner[i]].normal;
				normal[0] += weighted.x;
				normal[1] += weighted.y;
				normal[2] += weighted.z;
				out_indices.push_back(corner[i]);
			}
		}

	auto& first = vertices.front().position;
	vec3 low{first[0], first[1], first[2]}, high = low;
	for(auto& vertex: vertices) {
		low = {std::min(low.x, vertex.position[0]), std::min(low.y, vertex.position[1]), std::min(low.z, vertex.position[2])};
		high = {std::max(high.x, vertex.position[0]), std::max(high.y, vertex.position[1]), std::max(high.z, vertex.position[2])};
	}
	auto centre = (low + high) * 0.5f;
	auto extent = high - low;
	auto scale = 2.0f / std::max({extent.x, extent.y, extent.z});

	for(auto& vertex: vertices) {
		vertex.position[0] = (vertex.position[0] - centre.x) * scale;
		vertex.position[1] = (vertex.position[1] - centre.y) * scale;
		vertex.position[2] = (vertex.position[2] - centre.z) * scale;

		auto normal = normalize(vec3{vertex.normal[0], vertex.normal[1], vertex.normal[2]});
		vertex.normal[0] = normal.x;
		vertex.normal[1] = normal.y;
		vertex.normal[2] = normal.z;
	}

	std::println("teapot: {} vertices, {} triangles", vertices.size(), out_indices.size() / 3);
	return vertices;
}

}
