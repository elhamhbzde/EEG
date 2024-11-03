import datetime

patients = []
for i in range(5):
    print(f"Enter data for patient {i+1}:")
    first_name = input("First Name: ").upper()
    last_name = input("Last Name: ").upper()
    birth_year = int(input("Birth Year (YYYY): "))
    height = float(input("Height (m): "))
    weight = float(input("Weight (kg): "))
    previous_hospitalization = input("Previous Hospitalization (yes/no): ")
    blood_pressure = input("Blood Pressure: ")
    number_of_children = int(input("Number of Children: "))

    # Calculating age
    current_year = datetime.datetime.now().year
    age = current_year - birth_year

    # Calculating BMI
    bmi = weight / (height ** 2)

    patient = {
        'Name': f"{first_name} {last_name}",
        'Age': age,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'Previous Hospitalization': previous_hospitalization,
        'Blood Pressure': blood_pressure,
        'Number of Children': number_of_children
    }
    
    patients.append(patient)
    print(f"Patient {first_name} {last_name}'s data saved successfully.\n")


print(patients)