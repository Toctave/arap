#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <cstdio>
#include <iostream>

using igl::opengl::glfw::Viewer;

bool mouse_move(Viewer& viewer, int x, int y) {
    printf("%d %d\n", x, y);

    return false;
}

int main(int argc, char *argv[])
{
    // Inline mesh of a cube
    const Eigen::MatrixXd V= (Eigen::MatrixXd(8,3)<<
			      0.0,0.0,0.0,
			      0.0,0.0,1.0,
			      0.0,1.0,0.0,
			      0.0,1.0,1.0,
			      1.0,0.0,0.0,
			      1.0,0.0,1.0,
			      1.0,1.0,0.0,
			      1.0,1.0,1.0).finished();
    const Eigen::MatrixXi F = (Eigen::MatrixXi(12,3)<<
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

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    
    Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(),3,1);

    viewer.callback_mouse_move = 
	[&V,&F,&C](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		int fid;
		Eigen::Vector3f bc;
		// Cast a ray in the view direction starting from the mouse position
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;
		if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y),
					    viewer.core().view,
					    viewer.core().proj,
					    viewer.core().viewport,
					    V,
					    F,
					    fid,
					    bc))
		{
		    Eigen::MatrixXd face_points(3, 3);

		    face_points.row(0) = V.row(F(fid, 0));
		    face_points.row(1) = V.row(F(fid, 1));
		    face_points.row(2) = V.row(F(fid, 2));

		    Eigen::MatrixXd point =
			bc(0) * face_points.row(0) +
			bc(1) * face_points.row(1) +
			bc(2) * face_points.row(2);
		    
		    viewer.data().set_points(point, C);

		    return true;
		}
		return false;
	    };

    
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
