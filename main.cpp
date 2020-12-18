#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/per_vertex_normals.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>

#include <thread>

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
    Eigen::Vector3d color;
    switch(g % 6) {
    case 0:
	color = Eigen::Vector3d(26, 153, 136);
	break;
    case 1:
	color = Eigen::Vector3d(235,86,0);
	break;
    case 2:
	color = Eigen::Vector3d(183, 36, 92);
	break;
    case 3:
	color = Eigen::Vector3d(243,183,0);
	break;
    case 4:
	color = Eigen::Vector3d(55,50,62);
	break;
    case 5:
	color = Eigen::Vector3d(52,89,149);
	break;
    }

    Eigen::Vector3d white(255, 255, 255);
    for (int i = 0; i < g / 6; i++) {
	color = white - (white - color) * .75;
    }
    return color / 255;
}

bool load_model(const std::string& model_name, Mesh& mesh) {
    if (hasEnding(model_name, ".off")) {
	igl::readOFF(model_name, mesh.V, mesh.F);
    } else if (hasEnding(model_name, ".obj")) {
	igl::readOBJ(model_name, mesh.V, mesh.F);
    } else {
	return false;
    }

    igl::per_vertex_normals(mesh.V, mesh.F, mesh.N);

    return true;
}

void benchmark(const std::string& model_name, int iterations) {
    using namespace std::chrono;
    Mesh mesh;
    LaplacianSystem system;

    std::vector<FixedVertex> fixed_vertices = {
	{0, 0},
	{1, 0},
	{2, 0},
	{3, 0},
    };
    
    if (!load_model(model_name, mesh)) {
	return;
    }

    system_init(system, &mesh, 0.);
    if (!system_bind(system, fixed_vertices)) {
	return;
    }
    
    auto t0 = high_resolution_clock::now();

    system_solve(system, iterations);

    auto t1 = high_resolution_clock::now();

    duration<double> elapsed(t1 - t0);

    std::cout << model_name << ", " << mesh.V.rows() << ", " << elapsed.count() / iterations << std::endl;
}

void solve_loop(LaplacianSystem* system) {
    while (true) {
	system_iterate(*system);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
	std::cerr << "Usage : ./example <model file> <3 or more fixed indices, separated in groups by commas>\n";
	return 1;
    }

    Mesh mesh;
    if (!load_model(argv[1], mesh)) {
	std::cerr << "Could not load model." << std::endl;
	return 1;
    }

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
    
    system_init(system, &mesh, 0.002);

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

    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    Eigen::MatrixXd colors(mesh.V.rows(), 3);
    for (int i = 0; i < colors.rows(); i++) {
	colors.row(i) = Eigen::Vector3d(170,170,170) / 255.;
    }

    viewer.callback_pre_draw =
	[&system, &mesh, &t0, &colors](Viewer& viewer) -> bool
	    {
		auto t1 = high_resolution_clock::now();
		duration<double> elapsed(t1 - t0);

		double iterations_per_second = system.iterations / elapsed.count();

		std::cout << iterations_per_second << " iterations per second\n";
		
		if (system.mesh_access.try_lock()) {
		    viewer.data().set_vertices(mesh.V);

		    igl::per_vertex_normals(mesh.V, mesh.F, mesh.N);
		    viewer.data().set_normals(mesh.N);
		    
		    system.mesh_access.unlock();
		}
		
		return false;
	    };

    viewer.core().is_animating = true;
    viewer.core().background_color = Eigen::Vector4f(233,237,238, 255) / 255.0f;
    
    std::cout << argv[1] << " : " << mesh.V.rows() << " vertices.\n";
    if (!system_bind(system, fixed_vertices)) {
    	std::cerr << "Failed to bind mesh\n" << std::endl;
    	return 1;
    }

    viewer.data().set_points(highlighted_points, highlighted_colors);
    viewer.data().set_mesh(mesh.V, mesh.F);
    viewer.data().set_colors(colors);
    viewer.data().set_face_based(false);

    std::thread solver_thread(solve_loop, &system);
    
    viewer.launch();
}
