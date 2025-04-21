variable_d = 100
variable_u = 0
for i in range(1000000000):
    variable_d -= 1/1000000000
    variable_u += 1/1000000000
print(variable_d, variable_u)
