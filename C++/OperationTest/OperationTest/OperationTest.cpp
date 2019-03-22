




#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
	int size = 512;
	int size_r = ((size + 31)>>5);  // equal to 512/2^5, 31 to complement the size
	std::cout << size_r << std::endl;
	std::cout << size << std::endl;
}
