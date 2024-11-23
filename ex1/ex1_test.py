from ex1 import main
import os

# Define the directory path
directory_path = 'Test/combined_takes'

# Get a list of all files in the directory
files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

output = """(28, 29)
(28, 29)
(35, 36)
(35, 36)
(57, 58)
(57, 58)
(71, 72)
(71, 72)
(31, 32)
(31, 32)
(35, 36)
(35, 36)
(90, 91)
(90, 91)
(124, 125)
(124, 125)
(37, 38)
(37, 38)
(30, 31)
(30, 31)
(33, 34)
(33, 34)
(9, 10)
(9, 10)""".splitlines()

index = 0
error_count = 0
for file in files:
    if output[index] != str(main(file,1)):
        print("Error, result for file:'",file,"' with type:",1,"should be:",output[index],"but yours is:",str(main(file,1)))
        error_count += 1
    index+=1
    if output[index] != str(main(file,1)):
        print("Error, result for file:'",file,"' with type:",2,"should be:",output[index],"but yours is:",str(main(file,2)))
        error_count += 1
    index+=1

print("You have",error_count,"Erros")

