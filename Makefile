build: build-release

build-release:                                                                                     
	 mkdir -p build                                                                         
	 cd build && cmake .. -DCMAKE_BUILD_TYPE=Release                                        
	 make -C build -j $(nproc)                                                              
																																													
unit-test: build                                                                                
	 make -C build munet_tests -j $(nproc)                                                  
	 ./build/munet_tests                                                                    

py-test: build
	cd tests && python3 test_python.py

format:                                                                                    
	 find src -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\)' -exec clang-format -style=file -i {} \;                                                                                   
	 find src -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\|txt\|md|\sh|\)' -exec sed -i 's/[[:space:]]*$$//' {} \;                                                                 
	 find tests -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\)' -exec clang-format -style=file -i {} \;                                                                                   
	 find tests -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\|txt\|md|\sh|\)' -exec sed -i 's/[[:space:]]*$$//' {} \;                                                                 
	 find demos -regex '.*\.\(py\)' -exec black {} \;                                       
	 find demos -regex '.*\.\(py\)' -exec sed -i 's/[[:space:]]*$$//' {} \;                 
																																													
doc:                                                                                      
	 mkdir -p docs                                                                          
	 cd build && pdoc ./munet -o ../docs                                   

PHONY: all build build-release unit-test py-test perf-test format docs docker-build

perf-test: build-release
	 MUNET_RUN_PERF_TESTS=1 ./build/munet_tests --gtest_filter=PerformanceTest.*



docker-build:
	 ./tools/build_in_docker.sh
