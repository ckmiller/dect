
# tool settings
CXX    = clang++
CFLAGS = -I/opt/local/include/eigen3 -std=c++11

ifdef DEBUG
CFLAGS += -O0 -g -DDEBUG
else
CFLAGS += -O3 -fasm-blocks -ffast-math -funroll-loops -fstrict-aliasing -DEIGEN_NO_DEBUG
endif

# build products
CPP_FILES = $(shell ls *.cpp)
OBJ_FILES = $(CPP_FILES:%.cpp=%.o)

PROD = test

all: release
	
release:
	@make $(PROD)
	
debug:
	@make DEBUG=1 $(PROD)

%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

$(PROD) : $(OBJ_FILES)
	$(CXX) $(LDFLAGS) $^ -o $(PROD)

clean:
	rm -f *.o $(PROD)

