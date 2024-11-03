from datetime import datetime

# Sample users and their passwords
users_passwords = {
    'user1': 'pass1',
    'user2': 'pass2',
    # Additional users can be added here
}

# Store patient data
patient_data = []

# Input data for 10 patients
for i in range(10):
    name = input(f"Enter patient {i+1} name: ")
    dob = input("Enter DOB (DD/MM/YYYY): ")
    hospitalization_history = input("Any hospitalization history? ")

    # Calculate age from DOB
    dob_date = datetime.strptime(dob, "%d/%m/%Y")
    today = datetime.today()
    age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))

    patient_info = {
        'Name': name,
        'DOB': dob,
        'Age': age,
        'Hospitalization History': hospitalization_history
    }
    patient_data.append(patient_info)

# User authentication with 4 attempts
for attempt in range(1, 5):
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    if users_passwords.get(username) == password:
        user_name = input("Enter your name: ")
        if any(patient['Name'] == user_name for patient in patient_data):
            print("Your name is registered in the system successfully.")
        else:
            print("Your name is not recorded in the system.")
        break
    else:
        if attempt == 4:
            print("The system is locked. You don't have access to the information.")
        else:
            print("Incorrect password. Try again.")


print(patient_data)

