/*
*
*  Test for boost!
*
*
*/

#include <iostream>
#include "boost/thread.hpp"

using namespace std;

void zthread()
{
	cout << "hello " << endl;
}
int main(int argc, char *argv[])
{
	boost::function<void()> f(zthread);
	boost::thread t(f);
	t.join();
	cout << " thread is over " << endl;


	return 0;
}