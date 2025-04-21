#include "stdio.h"

void err_1()
{
    printf("Error 1");
    float a, b, f;
    a = 123456789;
    b = 123456788;     
    f = a - b;  
    printf("Result: %f\n", f);
}

void err_2()
{
    printf("Error 2");
    float a;
    double b, f;
    a = 123456789.123457;
    b = 123456789.123457;     
    f = a - b;  
    printf("Result: %f\n", f);
}

void err_3()
{
    printf("Error 3");
    printf("%f\n", 0.6000006 + 0.09999994);
    float a, b, c;
    a = 1;
    b = 3;
    c = a / b;
    c = c - 1.0f/3;
    printf("Result: %f\n", c);
}

void err_4()
{
    printf("Error 4");
    float a;
    float c;
    long n;
    c = 300;
    a = 0.00001;
    for (n = 1; n < 10000000; n++)
        c = c - a;
    printf("Result: %f\n", c);
}

int main(int argc, char *argv[])
{
    err_1();
    err_2();
    err_3();
    err_4();
    return 0;
}