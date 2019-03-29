

"""
 Test for  while operations
 a: zhong
 date:2019-3-26

"""

# message = input("Repeat back to you: ")
# print(message)
# a = 1 + int(message)
# print(a)
#
# print( 5 % 2)
# print( 4 % 2)


prompt = " Please input your data: "
message = ""
# while message != 'quit':
#     message = input(prompt)
#     if message != 'quit':
#         print(message)

# active = True
# while active:
#     message = input(prompt)
#     if message == 'quit':
#         active = False
#     else:
#         print(message)

# while True:
#     city = input(prompt)
#     if city == 'quit':
#         break
#     else:
#         print("I'd love to go to " + city.title() + "!")


# u_users = ['alice', 'brian', 'candace']
# c_users = []
#
# while u_users:
#     u_user = u_users.pop()
#     print(u_user.title())
#     c_users.append(u_user)
#
# for c_user in c_users:               # list don't have items; just dict?
#     print(c_user.title())

# pets = ['dog', 'cat', 'dog', 'goldfish', 'cat', 'rabbit', 'cat']
# print(pets)
# while 'cat' in pets:
#     pets.remove('cat')
# print(pets)


# def greet_user():
#     print("Hello")
#
# greet_user()


# def get_formatted_name(first_name, last_name):
#     """返回整洁的姓名"""
#     full_name = first_name + ' ' + last_name
#     return full_name.title()
#     # 这是一个无限循环!
# while True:
#     print("\nPlease tell me your name:")
#     f_name = input("First name: ")
#     if f_name == 'q':
#         break
#
#     l_name = input("Last name: ")
#     if l_name == 'q':
#         break
#
#     formatted_name = get_formatted_name(f_name, l_name)
#     print("\nHello, " + formatted_name + "!")


class Car():
    """ """
    def __init__(self, make, model, year):
        """  """
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        """  """
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name

    def read_odometer(self):
        """ """
        print(str(self.odometer_reading) + " mile on it !")

    def update_odometer(self, mileage):
        """  """
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print(" You can't roll back a odometer !")

    def increment_odometer(self, miles):
        """ """
        self.odometer_reading += miles

my_new_car = Car('audi', 'a4', 2016)
print(my_new_car.get_descriptive_name())
# my_new_car.odometer_reading = 23
my_new_car.update_odometer(23)
my_new_car.update_odometer(12)
my_new_car.increment_odometer(200)
my_new_car.read_odometer()

class ElectricCar(Car):
    """ """
    def __init__(self, make, model, year):
        """  """
        super().__init__(make, model, year)
        self.battery_size = 70

    def describe_battery(self):
        """  """
        print(str(self.battery_size))

my_tesla = ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())
my_tesla.describe_battery()









