#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

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

Eigen::Index closest_point(Eigen::Vector2f mouse,
			   const Mesh& mesh, const std::vector<Eigen::Index>& group,
			   Eigen::Matrix4f view, Eigen::Matrix4f proj,
			   Eigen::Vector2f point_size) {
    float closest = -1;
    Eigen::Index chosen = -1;

    float threshold = point_size.squaredNorm();

    Eigen::Vector4f p;
    p(3) = 1;
    for (int i = 0; i < group.size(); i++) {
	p.block<3, 1>(0, 0) = mesh.V.row(group[i]).cast<float>();
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

void update_group(const Eigen::MatrixXd& V, const std::vector<Eigen::Index>& group, Eigen::MatrixXd& group_pos) {
    for (int i = 0; i < group.size(); i++) {
	group_pos.row(i) = V.row(group[i]);
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

int main(int argc, char *argv[])
{
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F0;

    if (argc < 5) {
	std::cerr << "Usage : ./example <model file> <3 or more fixed indices>\n";
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

    Eigen::MatrixXd highlighted_points;
    Eigen::MatrixXd highlighted_colors;

    struct {
	Eigen::Index selected = -1;
	bool down = false;
	Eigen::Vector4f last_pos;
    } mouse;

    LaplacianSystem system;
    std::vector<Eigen::Index> fixed(argc - 2);

    system_init(system, &mesh);

    viewer.callback_mouse_move = 
	[&fixed, &system, &highlighted_colors, &highlighted_points, &mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		if (mouse.down && mouse.selected >= 0) {
		    Eigen::Vector4f unprojected_mouse = unproject_mouse(viewer, system.mesh->V.row(mouse.selected).cast<float>());
		    Eigen::Vector4f mouse_delta = unprojected_mouse - mouse.last_pos;
		    mouse.last_pos = unprojected_mouse;

		    system.mesh->V.row(mouse.selected) += mouse_delta.block<3, 1>(0, 0).cast<double>();

		    update_group(system.mesh->V, fixed, highlighted_points);
		    viewer.data().set_points(highlighted_points, highlighted_colors);

		    return true;
		}

		return false;
	    };

    viewer.callback_mouse_down = 
	[&mouse, &system, &fixed](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		Eigen::Vector2f dimensions = viewer.core().viewport.block<2, 1>(2, 0);
		Eigen::Vector2f point_size = viewer.data().point_size / dimensions.array();
		
		mouse.down = true;

		Eigen::Index closest = closest_point(mouse_position(viewer),
						     *system.mesh, fixed,
						     viewer.core().view,
						     viewer.core().proj,
						     point_size);
		mouse.selected = closest;

		if (closest >= 0) {
		    mouse.last_pos = unproject_mouse(viewer, system.mesh->V.row(mouse.selected).cast<float>());
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

    highlighted_points.resize(argc - 2, 3);
    highlighted_colors.resize(argc - 2, 3);
    
    for (int i = 0; i < argc - 2; i++) {
	fixed[i] = atoi(argv[i + 2]);
	highlighted_colors.row(i) << 1, 0, 0;
	highlighted_points.row(i) = mesh.V.row(fixed[i]);
    }

    if (!system_bind(system, fixed)) {
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
