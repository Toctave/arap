#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <cstdio>
#include <iostream>

#include "arap.hpp"

using igl::opengl::glfw::Viewer;

int main(int argc, char *argv[])
{
    // Inline mesh of a cube
    const Eigen::MatrixXd V0 = (Eigen::MatrixXd(8,3)<<
			      0.0,0.0,0.0,
			      0.0,0.0,1.0,
			      0.0,1.0,0.0,
			      0.0,1.0,1.0,
			      1.0,0.0,0.0,
			      1.0,0.0,1.0,
			      1.0,1.0,0.0,
			      1.0,1.0,1.0).finished();
    const Eigen::MatrixXi F0 = (Eigen::MatrixXi(12,3)<<
			       1,7,5,
			       1,3,7,
			       1,4,3,
			       1,2,4,
			       3,8,7,
			       3,4,8,
			       5,7,8,
			       5,8,6,
			       1,5,6,
			       1,6,2,
			       2,6,8,
			       2,8,4).finished().array()-1;

    Mesh mesh{V0, F0};
    // auto weights = cotangent_weights(mesh);

    // Plot the mesh
    Viewer viewer;
    
    Eigen::MatrixXd C(1, 3);
    C << 1, 0, 1;

    int closest = -1;
    bool mouse_is_down = false;
    float last_mouse_x(0), last_mouse_y(0);

    viewer.callback_mouse_move = 
	[&mesh, &V0, &C, &closest, &mouse_is_down, &last_mouse_x, &last_mouse_y](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		int fid;
		Eigen::Vector3f bc;
		// Cast a ray in the view direction starting from the mouse position
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;

		if (mouse_is_down && closest >= 0) {
		    float mouse_dx = x - last_mouse_x;
		    float mouse_dy = y - last_mouse_y;
		    Eigen::MatrixXf m = viewer.core().view.inverse();

		    Eigen::Vector4f dmouse(mouse_dx, mouse_dy, 0, 0);
		    Eigen::Vector4f dmouse_world = m * dmouse;

		    float speed = .01;
		    mesh.V(closest, 0) += dmouse_world(0) * speed;
		    mesh.V(closest, 1) += dmouse_world(1) * speed;
		    mesh.V(closest, 2) += dmouse_world(2) * speed;

		    viewer.data().set_mesh(mesh.V, mesh.F);

		    last_mouse_x = x;
		    last_mouse_y = y;
		    
		    return true;
		}
		last_mouse_x = x;
		last_mouse_y = y;
		
		if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y),
					    viewer.core().view,
					    viewer.core().proj,
					    viewer.core().viewport,
					    mesh.V,
					    mesh.F,
					    fid,
					    bc))
		{
		    Eigen::MatrixXd face_points(3, 3);

		    face_points.row(0) = mesh.V.row(mesh.F(fid, 0));
		    face_points.row(1) = mesh.V.row(mesh.F(fid, 1));
		    face_points.row(2) = mesh.V.row(mesh.F(fid, 2));

		    Eigen::MatrixXd point =
			bc(0) * face_points.row(0) +
			bc(1) * face_points.row(1) +
			bc(2) * face_points.row(2);

		    if (bc(0) > bc(1)) {
			if (bc(0) > bc(2)) {
			    closest = mesh.F(fid, 0);
			} else {
			    closest = mesh.F(fid, 2);
			}
		    } else {
			if (bc(1) > bc(2)) {
			    closest = mesh.F(fid, 1);
			} else {
			    closest = mesh.F(fid, 2);
			}
		    }
		    
		    viewer.data().set_points(mesh.V.row(closest), C);
		    return true;
		} else {
		    viewer.data().clear_points();
		    closest = -1;
		}
		
		return false;
	    };

    viewer.callback_mouse_down = 
	[&closest, &mouse_is_down](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		printf("closest = %d\n", closest);
		mouse_is_down = true;
		return false;
	    };

    viewer.callback_mouse_up = 
	[&mesh, &V0, &closest, &mouse_is_down](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		mouse_is_down = false;

		return false;
	    };

    std::vector<int> s = swizzle_from(6, {1, 3, 4});
    for (int i = 0; i < 6; i++) {
	printf("%d ", s[i]);
    }
    printf("\n");
    
    viewer.data().set_mesh(mesh.V, mesh.F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
