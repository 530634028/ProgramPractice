

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


# class Car():
#     """ """
#     def __init__(self, make, model, year):
#         """  """
#         self.make = make
#         self.model = model
#         self.year = year
#         self.odometer_reading = 0
#
#     def get_descriptive_name(self):
#         """  """
#         long_name = str(self.year) + ' ' + self.make + ' ' + self.model
#         return long_name
#
#     def read_odometer(self):
#         """ """
#         print(str(self.odometer_reading) + " mile on it !")
#
#     def update_odometer(self, mileage):
#         """  """
#         if mileage >= self.odometer_reading:
#             self.odometer_reading = mileage
#         else:
#             print(" You can't roll back a odometer !")
#
#     def increment_odometer(self, miles):
#         """ """
#         self.odometer_reading += miles
#
# my_new_car = Car('audi', 'a4', 2016)
# print(my_new_car.get_descriptive_name())
# # my_new_car.odometer_reading = 23
# my_new_car.update_odometer(23)
# my_new_car.update_odometer(12)
# my_new_car.increment_odometer(200)
# my_new_car.read_odometer()
#
# class ElectricCar(Car):
#     """ """
#     def __init__(self, make, model, year):
#         """  """
#         super().__init__(make, model, year)
#         self.battery_size = 70
#
#     def describe_battery(self):
#         """  """
#         print(str(self.battery_size))
#
# my_tesla = ElectricCar('tesla', 'model s', 2016)
# print(my_tesla.get_descriptive_name())
# my_tesla.describe_battery()


# try:
#     print(5/0)
# except ZeroDivisionError:
#     with open('test.txt', 'w') as file_object:
#         file_object.write("You can't division zero! ")
#     print("You ")


# while True:
#     first_number = input("\nFirst number: ")
#     if first_number == 'q':
#         break
#     second_number = input("\nFirst number: ")
#     if second_number == 'q':
#         break
#     try:
#         answer = int(first_number) / int(second_number)
#     except ZeroDivisionError:
#         with open('test.txt', 'w') as file_object:
#             file_object.write("You can't division zero! ")
#         print("You ")
#     else:
#         print(answer)


# title = "Alice in Wonderland"
# a = title.split()
# print(a)


# def count_words(filename):
#     """  """
#     try:
#         with open(filename, 'r') as file_object:
#             contents = file_object.read()
#     except FileNotFoundError:
#         # msag = "Can't find the " + filename + " book!"
#         # print(msag)
#         pass
#     else:
#         words = contents.split()
#         num = len(words)
#         print("Number of words in the book is: " + str(num))
#
# filename = "test.txt"
# count_words(filename)


# import json
# numbers = [2, 3, 5, 7, 11, 13]
# filename = 'numbers.json'
# with open(filename, 'w') as f_obj:
#     json.dump(numbers, f_obj)
#
# with open(filename, 'r') as f_obj:
#     numbers_r = json.load(f_obj)
#
# print(numbers_r)


# import json
#
# def get_stored_username():
#     """  """
#     filename = 'username.json'
#     try:
#         with open(filename) as f_obj:
#             username = json.load(f_obj)
#     except FileNotFoundError:
#         return None
#     else:
#         return username
#
# def get_new_username():
#     """  """
#     username = input("What is your name? ")
#     filename = 'username.json'
#     with open(filename, 'w') as f_obj:
#         json.dump(username, f_obj)
#     return username
#
# def greet_user():
#     """  """
#     username = get_stored_username()
#     if username:
#         print("Welcome back, " + username + "!")
#     else:
#         username = get_new_username()
#         print("We'll remember you when you come back, " + username + "!")
#
#
# greet_user()


""" function test """
# # def get_formatted_name(first, last):
# #     """  """
# #     full_name = first + ' ' + last
# #     return full_name.title()
#
# # def get_formatted_name(first, middle, last):
# #     """  """
# #     full_name = first + ' ' + middle + ' ' + last
# #     return full_name.title()
#
# def get_formatted_name(first, last, middle=''):
#     """  """
#     if middle:
#         full_name = first + ' ' + middle + ' ' + last
#     else:
#         full_name = first + ' ' + last
#     return full_name.title()
#
# # while True:
# #     first = input("\nfirst: ")
# #     if first == 'q':
# #         break
# #
# #     last = input("\nfirst: ")
# #     if last == 'q':
# #         break
# #
# #     formatted_name = get_formatted_name(first, last)
# #     print("\tNeatly formatted name: " + formatted_name + '.')
#
# import unittest
# class NamesTestCase(unittest.TestCase):
#     """  """
#     def test_first_last_name(self):
#         """  """
#         formatted_name = get_formatted_name('janis', 'joplin')
#         self.assertEqual(formatted_name, 'Janis Joplin')
#
#     def test_first_last_middle_name(self):
#         formatted_name = get_formatted_name('wolfgang', 'mozart', 'amadeus')
#         self.assertEqual(formatted_name, 'Wolfgang Amadeus Mozart')
#
# unittest.main()


""" class test """
class AnonymousSurvey():
    """收集匿名调查问卷的答案"""
    def __init__(self, question):
        """存储一个问题， 并为存储答案做准备"""
        self.question = question
        self.responses = []
    def show_question(self):
        """显示调查问卷"""
        print(question)
    def store_response(self, new_response):
        """存储单份调查答卷"""
        self.responses.append(new_response)
    def show_results(self):
        """显示收集到的所有答卷"""
        print("Survey results:")
        for response in responses:
            print('- ' + response)

import unittest
class TestAnonymousSurvey(unittest.TestCase):
    """针对AnonymousSurvey类的测试"""
    def setUp(self):
        """
        创建一个调查对象和一组答案， 供使用的测试方法使用
        """
        question = "What language did you first learn to speak?"
        self.my_survey = AnonymousSurvey(question)
        self.responses = ['English', 'Spanish', 'Mandarin']
    def test_store_single_response(self):
        """测试单个答案会被妥善地存储"""
        self.my_survey.store_response(self.responses[0])
        self.assertIn(self.responses[0], self.my_survey.responses)
    def test_store_three_responses(self):
        """测试三个答案会被妥善地存储"""
        for response in self.responses:
            self.my_survey.store_response(response)
        for response in self.responses:
            self.assertIn(response, self.my_survey.responses)
unittest.main()






