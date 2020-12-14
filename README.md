# As Rigid As Possible Surface deformation

An implementation of ARAP deformation using libigl. 

## Compile

libigl should be installed in the parent folder of where this repository has been cloned. 

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build .

This should find and build the dependencies and create an `example` binary.

## Run

From within the `build` directory issue one of those commands, or any variant thereof :

    ./example ../data/Meshes_ARAP_SA2007/cactus_small.off 504 , 266 276 277 288 289 290 291 292 300 304 305 306 307 308 314 315 331 332 348 350 351 353 354 355 356 357 358 576 577 583 584 587 588 589 594 595 596 602 603 604 607 608 609 610 611 612 613 614 615 616 617 618 619
	./example ../data/spot.obj 284 , 289 , 572 , 577 , 1239 , 1280 , 2369 , 1448
	./example ../data/armadillo.obj 304 , 1346 , 4600 , 9513 , 13411 , 18151 , 29237 , 30570 , 32747 , 33214 , 33495 , 36150 , 38168 , 38675 , 40000


