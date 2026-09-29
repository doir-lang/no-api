// The implementation halves of the two header-only asset loaders the tests use, in one
// translation unit so that neither is compiled twice and neither test has to remember which macro
// turns which one on. 1_texture only needs stb_image; 2_pbr needs both.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
