#include <iostream>

using namespace std;

int main()
{

    double x = 0.2;
    for (int i = 0; i < 40; i++)
    {
        x -= 0.01;
        if (x == 0)
            cout << "ok";
    }
}
