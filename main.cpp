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

Eigen::Index select_point(float x, float y,
			  const Mesh& mesh, const std::vector<Eigen::Index>& group,
			  Eigen::Matrix4f view, Eigen::Matrix4f proj,
			  Eigen::Vector4f viewport, float point_size) {
    Eigen::Vector2f mouse(x, y);
    float closest = -1;
    float threshold = point_size * point_size / 4;
    Eigen::Index chosen = -1;

    Eigen::Vector4f p;
    p(3) = 1;
    for (int i = 0; i < group.size(); i++) {
	p.block<3, 1>(0, 0) = mesh.V.row(group[i]).cast<float>();
	Eigen::Vector4f projected = (proj * view * p).transpose();
	projected /= projected(3);

	Eigen::Vector2f projected_pixels = (projected.block<2, 1>(0, 0).array() + 1.) * viewport.block<2, 1>(2, 0).array() / 2.;
	
	if ((projected_pixels - mouse).squaredNorm() <= threshold
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
	int selected = -1;
	bool down = false;
	bool dragging = false;
	float last_x;
	float last_y;
    } mouse = {-1, false, 0, 0};

    LaplacianSystem system;
    std::vector<Eigen::Index> fixed(argc - 2);

    system_init(system, &mesh);

    viewer.callback_mouse_move = 
	[&fixed, &mesh, &system, &V0, &highlighted_colors, &highlighted_points, &mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		int fid;
		Eigen::Vector3f bc;
		// Cast a ray in the view direction starting from the mouse position
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;

		Eigen::Index closest = select_point(x, y,
				       mesh, fixed,
				       viewer.core().view,
				       viewer.core().proj,
				       viewer.core().viewport,
				       viewer.data().point_size);

		bool ignore_camera = false;

		if (mouse.down && !mouse.dragging && closest >= 0) {
		    mouse.dragging = true;
		    mouse.selected = closest;
		}

		if (mouse.dragging) {
		    float mouse_dx = x - mouse.last_x;
		    float mouse_dy = y - mouse.last_y;
		    Eigen::MatrixXf m = viewer.core().view.inverse();

		    Eigen::Vector4f dmouse(mouse_dx, mouse_dy, 0, 0);
		    Eigen::Vector4f dmouse_world = m * dmouse;

		    float speed = .01;
		    mesh.V(mouse.selected, 0) += dmouse_world(0) * speed;
		    mesh.V(mouse.selected, 1) += dmouse_world(1) * speed;
		    mesh.V(mouse.selected, 2) += dmouse_world(2) * speed;

		    system_solve(system);
		    
		    viewer.data().set_mesh(mesh.V, mesh.F);
		    update_group(mesh.V, fixed, highlighted_points);
		    viewer.data().set_points(highlighted_points, highlighted_colors);
			
		    ignore_camera = true;
		}
		mouse.last_x = x;
		mouse.last_y = y;

		return ignore_camera;
	    };

    viewer.callback_mouse_down = 
	[&mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		mouse.down = true;
		return false;
	    };

    viewer.callback_mouse_up = 
	[&mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		mouse.down = false;
		mouse.dragging = false;

		return false;
	    };

    highlighted_points.resize(argc - 1, 3);
    highlighted_colors.resize(argc - 1, 3);
    
    for (int i = 0; i < argc - 2; i++) {
	fixed[i] = atoi(argv[i + 2]);
	highlighted_colors.row(i) << 1, 0, 0;
	highlighted_points.row(i) = mesh.V.row(fixed[i]);
    }

    if (!system_bind(system, fixed)) {
    	std::cerr << "Failed to bind mesh\n" << std::endl;
    	return 1;
    }

    // system_solve(system);

    std::cout << viewer.data().point_size << std::endl;
    viewer.data().set_points(highlighted_points, highlighted_colors);

    viewer.data().set_mesh(mesh.V, mesh.F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
