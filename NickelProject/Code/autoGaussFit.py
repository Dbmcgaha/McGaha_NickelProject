# This program keeps the user from analyzing all of the files one at a time

import subprocess

def run_program():
    
    # Nickel
    print('NICKEL3 DATA')
    for i in range(0, 17):
        print(f"Iteration {i}:")

        # Define the input sequence as a multi-line string with the iteration number (i)
        input_sequence = f"""nickel3
    {i}
    0
    128
    500"""

        # Execute the program.py with the updated input sequence
        subprocess.run(["python", "gaussFit.py"], input=input_sequence, text=True)
    
    print('NICKEL6 DATA')
    for i in range(0, 13):
        print(f"Iteration {i}:")

        # Define the input sequence as a multi-line string with the iteration number (i)
        input_sequence = f"""nickel6
    {i}
    0
    128
    500"""

        # Execute the program.py with the updated input sequence
        subprocess.run(["python", "gaussFit.py"], input=input_sequence, text=True)
    
    # Gold
    print('GOLD DATA')
    for i in range(0, 20):
        print(f"Iteration {i}:")

        # Define the input sequence as a multi-line string with the iteration number (i)
        input_sequence = f"""gold
    {i}
    0
    128
    500"""

        # Execute the program.py with the updated input sequence
        subprocess.run(["python", "gaussFit.py"], input=input_sequence, text=True)
    

if __name__ == "__main__":
    input('Press ENTER to start')
    run_program()