# Builds the LEMON reference solver. Point LEMON_DIR at a LEMON 1.3.x source
# tree; only its headers are used.
LEMON_DIR ?= $(HOME)/Development/lemon-1.3.1
CXXFLAGS ?= -O3 -march=native -DNDEBUG

mcf_ref: mcf_ref.cc
	$(CXX) -std=c++17 $(CXXFLAGS) -DLEMON_ONLY_TEMPLATES -Iinclude -I$(LEMON_DIR) -o $@ $<

# Same, with libstdc++ bounds checks on every container access.
mcf_ref_checked: mcf_ref.cc
	$(CXX) -std=c++17 -O1 -g -D_GLIBCXX_ASSERTIONS -DLEMON_ONLY_TEMPLATES -Iinclude -I$(LEMON_DIR) -o $@ $<

clean:
	rm -f mcf_ref mcf_ref_checked
