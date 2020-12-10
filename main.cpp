#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include <cstdlib>

#include "arap.hpp"

using igl::opengl::glfw::Viewer;

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

FixedVertex closest_point(Eigen::Vector2f mouse,
			   const Mesh& mesh, const std::vector<FixedVertex>& group,
			   Eigen::Matrix4f view, Eigen::Matrix4f proj,
			   Eigen::Vector2f point_size) {
    float closest = -1;
    FixedVertex chosen = {-1, 0};

    float threshold = point_size.squaredNorm();

    Eigen::Vector4f p;
    p(3) = 1;
    for (int i = 0; i < group.size(); i++) {
	p.block<3, 1>(0, 0) = mesh.V.row(group[i].index).cast<float>();
	Eigen::Vector4f projected = (proj * view * p);
	projected /= projected(3);

	// Eigen::Vector2f projected_pixels = (projected.block<2, 1>(0, 0).array() + 1.) * viewport.block<2, 1>(2, 0).array() / 2.;
	
	if ((projected.block<2, 1>(0, 0) - mouse).squaredNorm() <= threshold
	    && (closest < 0 || projected(2) < closest)) {
	    closest = projected(2);
	    chosen = group[i];
	}
    }

    return chosen;
}

void update_group(const Eigen::MatrixXd& V, const std::vector<FixedVertex>& group, Eigen::MatrixXd& group_pos) {
    for (int i = 0; i < group.size(); i++) {
	group_pos.row(i) = V.row(group[i].index);
    }
}

Eigen::Vector2f mouse_position(const Viewer& viewer) {
    Eigen::Vector2f dimensions = viewer.core().viewport.block<2, 1>(2, 0);
    Eigen::Vector2f mouse_pos(
	viewer.current_mouse_x,
	viewer.core().viewport(3) - viewer.current_mouse_y
	);

    mouse_pos.array() = 2. * mouse_pos.array() / dimensions.array() - 1.;

    return mouse_pos;
}

Eigen::Vector4f unproject_mouse(const Viewer& viewer, Eigen::Vector3f point) {
    Eigen::Vector2f mouse_pos = mouse_position(viewer);
    Eigen::Matrix4f viewproj = viewer.core().proj * viewer.core().view;

    Eigen::Vector4f point_homo;
    
    point_homo.block<3, 1>(0, 0) = point.cast<float>();
    point_homo(3) = 1.;
    Eigen::Vector4f projected_sel = viewproj * point_homo;
    projected_sel /= projected_sel(3);

    Eigen::Vector4f mouse_homo(mouse_pos(0), mouse_pos(1), projected_sel(2), 1.);
    Eigen::Vector4f unprojected_mouse = viewproj.inverse() * mouse_homo;
    unprojected_mouse /= unprojected_mouse(3);

    return unprojected_mouse;
}

bool compare_by_index(const FixedVertex& v1, const FixedVertex& v2) {
    return v1.index < v2.index;
}

Eigen::Vector3d group_color(size_t g) {
    switch(g % 6) {
    case 0:
	return Eigen::Vector3d(1, 0, 0);
    case 1:
	return Eigen::Vector3d(0, 1, 0);
    case 2:
	return Eigen::Vector3d(0, 0, 1);
    case 3:
	return Eigen::Vector3d(1, 1, 0);
    case 4:
	return Eigen::Vector3d(1, 0, 1);
    case 5:
	return Eigen::Vector3d(0, 1, 1);
    }
    return Eigen::Vector3d(0, 0, 0);
}

int main(int argc, char *argv[])
{
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F0;

    if (argc < 2) {
	std::cerr << "Usage : ./example <model file> <3 or more fixed indices, separated in groups by commas>\n";
	return 1;
    }
    
    std::string filename(argv[1]);
    if (hasEnding(filename, ".off")) {
	igl::readOFF(argv[1], V0, F0);
    } else if (hasEnding(filename, ".obj")) {
	igl::readOBJ(argv[1], V0, F0);
    } else {
	std::cerr << "Cannot recognize file " << filename << std::endl;
	return 1;
    }

    Mesh mesh{V0, F0};
    // auto weights = cotangent_weights(mesh);

    // Plot the mesh
    Viewer viewer;

    struct {
	FixedVertex selected = {.index = -1};
	bool down = false;
	Eigen::Vector4f last_pos;
    } mouse;

    LaplacianSystem system;
    std::vector<FixedVertex> fixed_vertices;

    size_t curgrp = 0;
    for (int i = 2; i < argc; i++) {
	if (argv[i][0] == ',') {
	    curgrp++;
	} else {
	    fixed_vertices.push_back({atoi(argv[i]), curgrp});
	}
    }

    std::sort(fixed_vertices.begin(), fixed_vertices.end(), compare_by_index);

    Eigen::MatrixXd highlighted_points(fixed_vertices.size(), 3);
    Eigen::MatrixXd highlighted_colors(fixed_vertices.size(), 3);

    for (size_t i = 0; i < fixed_vertices.size(); i++) {
	highlighted_colors.row(i) = group_color(fixed_vertices[i].group);
	highlighted_points.row(i) = mesh.V.row(fixed_vertices[i].index);
    }
    
    system_init(system, &mesh);

    viewer.callback_mouse_move = 
	[&fixed_vertices, &system, &highlighted_colors, &highlighted_points, &mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		if (mouse.down && mouse.selected.index >= 0) {
		    Eigen::Vector4f unprojected_mouse = unproject_mouse(viewer, system.mesh->V.row(mouse.selected.index).cast<float>());
		    Eigen::Vector4f mouse_delta = unprojected_mouse - mouse.last_pos;
		    mouse.last_pos = unprojected_mouse;

		    for (const auto& vertex : fixed_vertices) {
			if (vertex.group == mouse.selected.group) {
			    system.mesh->V.row(vertex.index) += mouse_delta.block<3, 1>(0, 0).cast<double>();
			}
		    }

		    update_group(system.mesh->V, fixed_vertices, highlighted_points);
		    viewer.data().set_points(highlighted_points, highlighted_colors);

		    return true;
		}

		return false;
	    };

    viewer.callback_mouse_down = 
	[&mouse, &system, &fixed_vertices](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		Eigen::Vector2f dimensions = viewer.core().viewport.block<2, 1>(2, 0);
		Eigen::Vector2f point_size = viewer.data().point_size / dimensions.array();
		
		mouse.down = true;

		FixedVertex closest = closest_point(mouse_position(viewer),
						     *system.mesh, fixed_vertices,
						     viewer.core().view,
						     viewer.core().proj,
						     point_size);
		mouse.selected = closest;

		if (closest.index >= 0) {
		    mouse.last_pos = unproject_mouse(viewer, system.mesh->V.row(mouse.selected.index).cast<float>());
		}
		
		return false;
	    };

    viewer.callback_mouse_up = 
	[&mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		mouse.down = false;

		return false;
	    };

    viewer.callback_pre_draw =
	[&system, &mesh](Viewer& viewer) -> bool
	    {
		system_iterate(system);
		viewer.data().set_vertices(mesh.V);
		return false;
	    };


    viewer.callback_key_down = 
	[&mesh, &mouse, &system](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)->bool
	    {
		if (key == 'U') { //marche aussi avecCtrl
		    mesh.V(0,0) += 1;
		    mouse.selected.index = 0;
		    viewer.data().set_mesh(mesh.V, mesh.F);
		    return true;
		}

		if (key == 'S') {
		    std::cout << std::endl << "Points: " << std::endl << mesh.V << std::endl;

		    system_iterate(system);
		    viewer.data().set_mesh(mesh.V, mesh.F);

		    std::cout << std::endl << "Points: " << std::endl << mesh.V << std::endl;
		    return true;
		}
		return false;
	    };
    
    viewer.core().is_animating = true;

    if (!system_bind(system, fixed_vertices)) {
    	std::cerr << "Failed to bind mesh\n" << std::endl;
    	return 1;
    }

    system_solve(system);

    std::cout << viewer.data().point_size << std::endl;
    viewer.data().set_points(highlighted_points, highlighted_colors);

    viewer.data().set_mesh(mesh.V, mesh.F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
