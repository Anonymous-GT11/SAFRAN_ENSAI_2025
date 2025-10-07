"""This file defines functions that aims to construct a map from a txt file with map compressor values"""

import numpy as np
import pandas as pd


def parse_file(filepath, lbs=True, write=False):
    """
    Function that parse a txt file and output a dataframe containing the information.
    Input : filepath = path for txt file that needs parsing. Must be in the format of the Ge10stg.txt file in the data folder
    Output : data = a dataframe containing the information of the txt file
    """

    # Open the file as a file_object
    with open(filepath, "r") as file_object:
        # Initialize list for the output
        data_beta = []
        data_n = []
        data_mass_flow = []
        data_efficiency = []
        data_pressure_ratio = []

        # Read first line of file_object
        line = file_object.readline()

        while line:  # while there are lines to be read in the document
            # Read next line
            line = file_object.readline()
            # If we reach the "Mass Flow" line, then begin to read and store values
            if line.strip() == "Mass Flow":
                # first get the beta column and then add it to the beta vector at each new N (only need to do it for Mass Flow ? or for each and then merge on beta ?)
                line = file_object.readline()
                values = line.split()
                # The N parameter is the first elements of the first line of each "paragraph" in the file
                current_n = values[0]
                data_n += [current_n] * len(values[1:])
                # Other data of the line are Mass Flow values
                data_mass_flow += values[1:]

                while (
                    line.strip() != "Efficiency"
                ):  # While we have not reached the Efficiency line, browse and store the values
                    values = line.split()
                    if (
                        len(values) == 1
                    ):  # If we arrive at the end of a paragraph, store the last value, read the next line, get the N parameter and the Mass Flow values and read the next line
                        data_mass_flow += values
                        data_n.append(current_n)
                        line = file_object.readline()
                        values = line.split()
                        while not values:  # If a line is empty, keep reading until a non empty line is found
                            line = file_object.readline()
                            values = line.split()

                        if line.strip() != "Efficiency":
                            current_n = values[0]
                            data_n += [current_n] * len(values[1:])
                            data_mass_flow += values[1:]
                            line = file_object.readline()

                    else:  # If we are within a paragraph, just read and store the values into the dedicated list
                        data_mass_flow += values
                        data_n += [current_n] * len(values)
                        line = file_object.readline()

                    if not values:
                        line = file_object.readline()
                        values = line.split()

            # Then proceed as before for the Efficiency and Pressure Ratio data
            if line.strip() == "Efficiency":
                line = file_object.readline()
                values = line.split()
                data_efficiency += values[1:]
                while line.strip() != "Pressure Ratio":
                    values = line.split()
                    if len(values) == 1:
                        data_efficiency += values
                        line = file_object.readline()
                        values = line.split()
                        while not values:
                            line = file_object.readline()
                            values = line.split()

                        if line.strip() != "Pressure Ratio":
                            data_efficiency += values[1:]
                            line = file_object.readline()

                    else:
                        data_efficiency += values
                        line = file_object.readline()

            if line.strip() == "Pressure Ratio":
                line = file_object.readline()
                values = line.split()
                data_pressure_ratio += values[1:]
                while line.strip() != "Surge Line":
                    values = line.split()
                    if len(values) == 1:
                        data_pressure_ratio += values
                        line = file_object.readline()
                        values = line.split()
                        while not values:
                            line = file_object.readline()
                            values = line.split()

                        if line.strip() != "Surge Line":
                            data_pressure_ratio += values[1:]
                            line = file_object.readline()

                    else:
                        data_pressure_ratio += values
                        line = file_object.readline()

    # Gather all the values into one dataframe
    data = pd.DataFrame(
        {
            "N": list(map(float, data_n)),
            "Mass Flow": list(map(float, data_mass_flow)),
            "Efficiency": list(map(float, data_efficiency)),
            "Pressure Ratio": list(map(float, data_pressure_ratio)),
        }
    )
    # Remove data corresponding to the first paragraph and an impossible value of "11.016" for the N parameter. This paragraph is useful if we use the beta parameter instead.
    data = data[data.N != 11.016].reset_index(drop=True)

    if lbs:
        data["Mass Flow Lbs"] = data["Mass Flow"]
        data["Mass Flow"] = data["Mass Flow Lbs"] * 0.45359

    if write:
        data.to_csv("./data/mapcompSafran.csv", index=False)
    # Return the dataframe
    return data


def parse_file_beta(filepath, lbs=True, write=False):
    """
    Function that parse a txt file and output a dataframe containing the information.
    Input : filepath = path for txt file that needs parsing. Must be in the format of the Ge10stg.txt file in the data folder
    Output : data = a dataframe containing the information of the txt file
    """

    # Open the file as a file_object
    with open(filepath, "r") as file_object:
        # Initialize list for the output
        data_beta = []
        data_n = []
        data_mass_flow = []
        data_efficiency = []
        data_pressure_ratio = []

        # Read first line of file_object
        line = file_object.readline()

        while line:  # while there are lines to be read in the document
            # Read next line
            line = file_object.readline()
            # If we reach the "Mass Flow" line, then begin to read and store values
            if line.strip() == "Mass Flow":
                beta = []
                # first get the beta column and then add it to the beta vector at each new N (only need to do it for Mass Flow ? or for each and then merge on beta ?)
                line = file_object.readline()
                values = line.split()
                # Construct the beta
                beta += values[1:]
                
                while(len(values) > 1):
                    line = file_object.readline()
                    values = line.split()
                    beta+=values


                line = file_object.readline()
                values = line.split()
                while not values:  # If a line is empty, keep reading until a non empty line is found
                    line = file_object.readline()
                    values = line.split()

                current_n = values[0]
                data_mass_flow +=values[1:]
                data_n += [current_n]*len(values[1:])
                line = file_object.readline()


                while (
                    line.strip() != "Efficiency"
                ):  # While we have not reached the Efficiency line, browse and store the values
                    values = line.split()
                    if (
                        len(values) == 1
                    ):  # If we arrive at the end of a paragraph, store the last value, read the next line, get the N parameter and the Mass Flow values and read the next line
                        data_mass_flow += values
                        data_n.append(current_n)
                        data_beta += beta
                        line = file_object.readline()
                        values = line.split()
                        while not values:  # If a line is empty, keep reading until a non empty line is found
                            line = file_object.readline()
                            values = line.split()

                        if line.strip() != "Efficiency":
                            current_n = values[0]
                            data_n += [current_n] * len(values[1:])
                            data_mass_flow += values[1:]
                            line = file_object.readline()

                    else:  # If we are within a paragraph, just read and store the values into the dedicated list
                        data_mass_flow += values
                        data_n += [current_n] * len(values)
                        line = file_object.readline()

                    if not values:
                        line = file_object.readline()
                        values = line.split()

            # Then proceed as before for the Efficiency and Pressure Ratio data
            if line.strip() == "Efficiency":
                line = file_object.readline()
                values = line.split()
                while(len(values)>1):
                    line = file_object.readline()
                    values = line.split()

                line = file_object.readline()
                values = line.split()
                data_efficiency += values[1:]
                line = file_object.readline()

                while line.strip() != "Pressure Ratio":
                    values = line.split()
                    if len(values) == 1:
                        data_efficiency += values
                        line = file_object.readline()
                        values = line.split()
                        while not values:
                            line = file_object.readline()
                            values = line.split()

                        if line.strip() != "Pressure Ratio":
                            data_efficiency += values[1:]
                            line = file_object.readline()

                    else:
                        data_efficiency += values
                        line = file_object.readline()

            if line.strip() == "Pressure Ratio":
                line = file_object.readline()
                values = line.split()
                while(len(values)>1):
                    line = file_object.readline()
                    values = line.split()

                line = file_object.readline()
                values = line.split()
                data_pressure_ratio += values[1:]
                line = file_object.readline()

                while line.strip() != "Surge Line":
                    values = line.split()
                    if len(values) == 1:
                        data_pressure_ratio += values
                        line = file_object.readline()
                        values = line.split()
                        while not values:
                            line = file_object.readline()
                            values = line.split()

                        if line.strip() != "Surge Line":
                            data_pressure_ratio += values[1:]
                            line = file_object.readline()

                    else:
                        data_pressure_ratio += values
                        line = file_object.readline()
    # Gather all the values into one dataframe
    data = pd.DataFrame(
        {   "beta": list(map(float, data_beta)),
            "N": list(map(float, data_n)),
            "Mass Flow": list(map(float, data_mass_flow)),
            "Efficiency": list(map(float, data_efficiency)),
            "Pressure Ratio": list(map(float, data_pressure_ratio)),
        }
    )


    if lbs:
        data["Mass Flow Lbs"] = data["Mass Flow"]
        data["Mass Flow"] = data["Mass Flow Lbs"] * 0.45359

    if write:
        data.to_csv("./data/mapcompSafran.csv", index=False)
    # Return the dataframe
    return data

def parse_file_simu(filepath, nline=2, lbs=False):
    with open(filepath, "r") as file_object:
        # Initialize list for the output
        data_n = []
        data_mass_flow = []
        data_efficiency = []
        data_pressure_ratio = []

        # Read first line of file_object
        line = file_object.readline()

        while line:  # while there are lines to be read in the document
            # Read next line
            line = file_object.readline()
            # If we reach the "Mass Flow" line, then begin to read and store values
            if line.strip() == "Mass Flow":
                for i in range(3):
                    line = file_object.readline()
                values = line.split()
                # The N parameter is the first elements of the first line of each "paragraph" in the file
                current_n = values[0]

                data_n += [current_n] * len(values[1:])
                # Other data of the line are Mass Flow values
                data_mass_flow += values[1:]

                line = file_object.readline()
                values = line.split()
                data_n += [current_n] * len(values)
                data_mass_flow += values

                line = file_object.readline()
                values = line.split()

                while (
                    line.strip() != "Efficiency"
                ):  # While we have not reached the Efficiency line, browse and store the values
                    current_n = values[0]
                    data_n += [current_n] * len(values[1:])
                    # Other data of the line are Mass Flow values
                    data_mass_flow += values[1:]

                    line = file_object.readline()
                    values = line.split()
                    data_n += [current_n] * len(values)
                    data_mass_flow += values

                    line = file_object.readline()
                    values = line.split()
                    while not values:
                        line = file_object.readline()
                        values = line.split()

            # Then proceed as before for the Efficiency and Pressure Ratio data
            if line.strip() == "Efficiency":
                for i in range(3):
                    line = file_object.readline()
                values = line.split()

                # Other data of the line are Mass Flow values
                data_efficiency += values[1:]

                line = file_object.readline()
                values = line.split()
                data_efficiency += values

                line = file_object.readline()
                values = line.split()

                while (
                    line.strip() != "Pressure Ratio"
                ):  # While we have not reached the Efficiency line, browse and store the values
                    # Other data of the line are Mass Flow values
                    data_efficiency += values[1:]

                    line = file_object.readline()
                    values = line.split()
                    data_efficiency += values

                    line = file_object.readline()
                    values = line.split()
                    while not values:
                        line = file_object.readline()
                        values = line.split()

            if line.strip() == "Pressure Ratio":
                for i in range(3):
                    line = file_object.readline()
                values = line.split()
                # The N parameter is the first elements of the first line of each "paragraph" in the file
                # Other data of the line are Mass Flow values
                data_pressure_ratio += values[1:]

                line = file_object.readline()
                values = line.split()
                data_pressure_ratio += values

                line = file_object.readline()
                values = line.split()

                while (
                    line.strip() != "Surge Line"
                ):  # While we have not reached the Efficiency line, browse and store the values
                    # Other data of the line are Mass Flow values
                    data_pressure_ratio += values[1:]

                    line = file_object.readline()
                    values = line.split()
                    data_pressure_ratio += values

                    line = file_object.readline()
                    values = line.split()
                    while not values:
                        line = file_object.readline()
                        values = line.split()

    # Gather all the values into one dataframe
    data = pd.DataFrame(
        {
            "N": list(map(float, data_n)),
            "Mass Flow": list(map(float, data_mass_flow)),
            "Efficiency": list(map(float, data_efficiency)),
            "Pressure Ratio": list(map(float, data_pressure_ratio)),
        }
    )
    # Remove data corresponding to the first paragraph and an impossible value of "11.016" for the N parameter. This paragraph is useful if we use the beta parameter instead.
    # data = data[data.N != 11.016].reset_index(drop = True)

    if lbs:
        data["Mass Flow Lbs"] = data["Mass Flow"]
        data["Mass Flow"] = data["Mass Flow Lbs"] * 0.45359

    # Return the dataframe
    return data
