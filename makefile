all:
	mpic++ -Wall -O3 -o dpd.x dpd.cpp
force_test:
	mpic++ -Wall -O3 -D FORCE_TEST -o dpd_force.x dpd.cpp
torque_test:
	mpic++ -Wall -O3 -D TORQUE_TEST -o dpd_torque.x dpd.cpp
dpd:
	mpic++ -Wall -O3 -o dpd.x dpd.cpp
temperature_test:
	mpic++ -Wall -O3 -D TEMPERATURE_TEST -o dpd_kbt.x dpd.cpp
ncutoff_test:
	mpic++ -Wall -O3 -D NCUTOFF_TEST -o dpd_nc.x dpd.cpp
viscosity:
	mpic++ -Wall -O3 -D DETERMINE_VISCOSITY -o dpd_visc.x dpd.cpp
