CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkantest: main.cpp
	g++ $(CFLAGS) -o vulkantest main.cpp $(LDFLAGS)

.PHONY: test clean dev

test: vulkantest
	./vulkantest

clean:
	rm -f vulkantest

dev:
	watchexec -e cpp,h make
