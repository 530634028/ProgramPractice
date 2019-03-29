

"""
 Test for traverse operations on list data struction
 a: zhong
 date:2019-3-18

"""

# magicians = ['alice', 'david', 'carolina']
# for magician in magicians:
#     # print(magician)
#     print(magician.title() + ", that was a great trick!")
#     print("I can't wait to see your next trick, " + magician.title() + ".\n")
#


# message = "hello"
#     print(message)  # IndentationError: unexpected indent


# numbers
# for value in range(1, 6):  # exclude the last number 5
#     print(value)

# numbers = list(range(1,6))
# print(numbers)
# even_numbers = list(range(2, 11, 2))
# print(even_numbers)


# squares = []
# for value in range(1,11):
#     # square = value**2
#     # squares.append(square)
#     squares.append(value**2)
#
# print(squares)


# digits = []
# for i in range(1,10):
#     digits.append(i)
#
# print(min(digits))
# print(max(digits))
# print(sum(digits))


# squares = [value**2 for value in range(1,11)]
# print(squares)


# players = ['charles', 'martina', 'michael', 'florence', 'eli']
# print(players[0:3])   # exclude element whose index is 3(remember)
# print(players[1:4])
# print(players[:4])    # start from first element of players list
# print(players[2:])    # end at the last element if players list
# print(players[-3:])   #
# for player in players[:3]:
#     print(player.title())
#
# playerTest = players[:]
# print(players)
# print(playerTest)
# playerTest = players    # attention for this form
# players.append("a")
# playerTest.append("b")
# print(players)
# print(playerTest)

# dimensions = (200, 30)        # tuple type
# print(dimensions[0])
# print(dimensions[1])
# # dimension[0] = 250
# for dimension in dimensions:
#     print(dimension)
# dimensions = (400, 100)
# for dimension in dimensions:
#     print(dimension)
#
# playerTmp = 'Martina'
# for player in players:
#     print(player == playerTmp.lower())
#     if player == playerTmp.lower() and playerTmp:#'martina':
#         print(player.upper())
#     else:
#         print(player.title())
#
# print(playerTmp)

# alien_0 = {
#     'color': 'greed',
#     'points': 5
# }
# print(alien_0['color'])
# print(alien_0['points'])
# print(alien_0)
#
# alien_0['x_position'] = 0
# alien_0['y_position'] = 25
# print(alien_0)
#
# del alien_0['points']
# print(alien_0)
#
#
# print("alien's color is: " +
#       alien_0['color'].title() +
#       '.')


# user_0 = {
# 'username': 'efermi',
# 'first': 'enrico',
# 'last': 'fermi',
# }
# for key, value in user_0.items():  # function items()
#     print("\nkey:" + key)
#     print("Value:" + str(value))
#
# for key in user_0.keys():          # function keys()
#     print(key.title())
# for key in user_0:
#     print(key.title())
#
# part = ['username']
# for key in user_0:
#     print(key.title())
#     if key in part:
#         print(" Hi " + key.title() +
#               ", I know you value is: " +
#               user_0[key].title() + "!")
#
# if 'erin' not in user_0.keys():
#     print(" Please add the erin !")


# favorite_languages = {
# 'jen': 'python',
# 'sarah': 'c',
# 'edward': 'ruby',
# 'phil': 'python',
# }
#
# for name in sorted(favorite_languages.keys()):
#     print(name.title() + ", thank you for taking the poll.")
#
# for languages in favorite_languages.values():
#     print(value.title())
#
# for language in set(favorite_languages.values()):
#     print(language.title())


# alien_0 = {'color': 'green', 'points': 5}
# alien_1 = {'color': 'yellow', 'points': 10}
# alien_2 = {'color': 'red', 'points': 15}
#
# aliens = [alien_0, alien_1, alien_2]
#
# for alien in aliens:
#     print(alien)

# aliens = []
# for alien_number in range(30):
#     new_alien = {'color': 'green', 'points': 5, 'speed': 'slow'}
#     aliens.append(new_alien)
#
# for alien in aliens[0:3]:
#     if alien['color'] == 'green':
#         alien['color'] = 'yellow'
#         alien['points'] = 10
#         alien['speed'] = 'medium'
#
# for alien in aliens[:5]:
#     print(alien)
# print("...")
# print("Total number of aliens: " + str(len(aliens)))


# pizza = {
#     'crust': 'thick',
#     'toppings': ['mushrooms', 'extra cheese']
# }
#
# print(pizza['crust'])
# for topping in pizza['toppings']:
#     print(topping)


users = {
'aeinstein': {
'first': 'albert',
'last': 'einstein',
'location': 'princeton',
},
'mcurie': {
'first': 'marie',
'last': 'curie',
'location': 'paris',
},
}

for usename, user_info in users.items():
    print(usename.title())
    print(user_info['first'].title() + " " + user_info['last'] +
          " " + user_info['location'])











