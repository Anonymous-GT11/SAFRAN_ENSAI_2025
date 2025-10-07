import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.path import Path
from scipy.interpolate import LinearNDInterpolator, griddata, interp1d

# Import homemade function
from performancemodel.parser_data import parse_file, parse_file_beta


pd.options.mode.chained_assignment = None  # default='warn'


#### This file contains function that helps compute different module for the TurboReactor simulator


##########################
# Approximate functions  #
##########################


def get_interpolation_withn(data, n):
    """Get interpolation functions for Pressure Ratio and Efficiency for a specific N value

    Parameters
    ----------
    data : dataframe
        The dataframe that contains the original data for computing interpolation.
    n : float
        New n value to interpolate

    Returns
    -------
    function_pr : function
        function that inputs a Mass Flow and output a pressure ratio from the compressor map
    function_eff : function
        function that inputs a Mass Flow and output an Efficiency from the compressor map
    """

    data_temp = data[data.N == n]
    x = data_temp["Mass Flow"]
    y = data_temp["Pressure Ratio"]
    z = data_temp["Efficiency"]
    function_pr = interp1d(
        x, y, fill_value="extrapolate"
    )  # Interpolate for Pressure Ratio

    function_eff = interp1d(
        x, z, fill_value="extrapolate"
    )  # Interpolate for Efficiency

    return function_pr, function_eff


def get_2d_interpolation(data, input_names, output_names):
    """Get 2-d interpolation functions for any 2-d inputs and 2-d outputs among (N, Mass Flow, Pressure Ratio and Efficiency)

    Parameters
    ----------
    data : dataframe
        The dataframe that contains the original data for computing interpolation.
    input_names : list of size 2
        List of the name of the inputs for the 2D interpolation function
    output_names : list of size 1 or 2
        List of the name of the outputs for the 2D interpolation function

    Returns
    -------
    function_out : function
        function that inputs variables values from input_names and outputs variables values from output_names from the compressor map

    """

    data = data.sort_values(input_names)
    fit_points = list(zip(data[input_names[0]], data[input_names[1]]))

    if len(output_names) > 1:  # If output_names contains 2 names
        values = list(zip(data[output_names[0]], data[output_names[1]]))
    else:
        values = data[output_names[0]]

    function_out = LinearNDInterpolator(fit_points, values)

    return function_out


def get_pol_approx(data, deg_pol):
    """Get 2-d interpolation functions for any 2-d inputs and 2-d outputs among (N, Mass Flow, Pressure Ratio and Efficiency)

    Parameters
    ----------
    data : dataframe
        The dataframe that contains the original data for computing interpolation.
    input_names : list of size 2
        List of the name of the inputs for the 2D interpolation function
    output_names : list of size 1 or 2
        List of the name of the outputs for the 2D interpolation function

    Returns
    -------
    function_out : function
        function that inputs variables values from input_names and outputs variables values from output_names from the compressor map

    """

    approx_pr = []
    approx_eff = []
    for n in data.N.unique():
        data_temp = data[data.N == n]
        x = data_temp["Mass Flow"]
        y = data_temp["Pressure Ratio"]
        z = data_temp["Efficiency"]
        approx = np.polyfit(x, y, deg_pol)

        approx_temp = 0
        for i in range(deg_pol + 1):
            approx_temp += approx[i] * x ** (deg_pol - i)

        approx_pr += list(approx_temp)

        approx = np.polyfit(x, z, deg_pol)
        approx_temp = 0
        for i in range(deg_pol + 1):
            approx_temp += approx[i] * x ** (deg_pol - i)
        approx_eff += list(approx_temp)

    data["Approx Pressure Ratio"] = approx_pr
    data["Approx Efficiency"] = approx_eff

    return data


def get_interpolation_from_file_withn(data, n):
    function_pr, function_eff = get_interpolation_withn(data, n)

    return function_pr, function_eff


def get_2dinterpolation_from_file(data, inputs, outputs):
    function_out = get_2d_interpolation(data, inputs, outputs)

    return function_out


def is_inside_quadrilateral_old(point, quad_vertices, draw=False, margin=0.0001):
    """
    Check if a point is inside a quadrilateral defined by four vertices.
    Parameters:

        - point: Tuple (x, y) of the point to check.

        - quad_vertices: List of four tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] defining the quadrilateral.

    Returns:

        - True if the point is inside the quadrilateral, False otherwise.
    """
    vertices = np.array(quad_vertices)

    if draw:
        for p1, p2 in zip(
            vertices, np.concatenate([vertices[1:, :], vertices[0, :].reshape(1, -1)])
        ):
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-")
            plt.plot(point[0], point[1], "bo")

    path = Path(vertices)

    return path.contains_point(point, radius=-margin)


def is_on_segment(p, v, w):
    """
    Check if p is on the segment [v,w]
    Parameters
            ----------
            p, v and w are points

            Returns
            -------
            dot_product <= 0 : binary
                binary output, value is 1 if p is on the [v,w] segment and 0 otherwise
    """
    # Compute vectors
    v_w = np.subtract(w, v)
    v_p = np.subtract(p, v)
    w_p = np.subtract(p, w)

    # Check if p is colineaire with v and w
    collinear = np.cross(v_w, v_p) == 0

    if not collinear:
        return False

    # Check if p is between v and w
    dot_product = np.dot(v_p, w_p)
    return dot_product <= 0


def is_inside_quadrilateral(point, quad_vertices, draw=False, margin=0.0001):
    """
    Check if a point is inside a quadrilateral defined by four vertices.

    Parameters:
        point (tuple): Tuple (x, y) of the point to check.
        quad_vertices (list): List of four tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] defining the quadrilateral.
        draw (bool): Whether to draw the quadrilateral and the point. Defaults to False.
        margin (float): Margin for the point-in-polygon check. Defaults to 0.0001.

    Returns:
        bool: True if the point is inside the quadrilateral or on its boundary, False otherwise.
    """
    vertices = np.array(quad_vertices)
    point = np.array(point)

    if draw:
        for p1, p2 in zip(vertices, np.concatenate([vertices[1:], vertices[:1]])):
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-")
        plt.plot(point[0], point[1], "bo")
        plt.show()

    # Vérifier si le point est à l'intérieur du quadrilatère
    path = Path(vertices)
    if path.contains_point(point, radius=-margin):
        return True

    # Vérifier si le point est sur une des arêtes du quadrilatère
    for v, w in zip(vertices, np.concatenate([vertices[1:], vertices[:1]])):
        if is_on_segment(point, v, w):
            return True

    return False


def search_point(x, y, name_mf="Mass Flow", df=None):
    """
    Get the coordinate of the the quadrilateral in which (x,y) is

    Parameters:
        x : float, first coordinate of the point to check
        y : float, second coordinate of the point to check
        mf : str, name of the second feature (first one is always N)
        df : Pandas DataFrame, dataframe with the data

    Returns:
        coords: returns the 4 coordinates of the quadrilateral the point (x,y) belongs to, *returns False if no coords were found i.e. the point were not in any quadrilateral.
    """

    listNunique = df["N"].unique()
    coords = np.nan

    for i in np.arange(len(listNunique) - 1):

        indexN = listNunique[i] == df["N"]
        indexN_plusone = listNunique[i + 1] == df["N"]

        listMFunique = np.array(df.loc[indexN, name_mf])
        listMFunique_plusone = np.array(df.loc[indexN_plusone, name_mf])



        for j in np.arange(14):
            point_1 = [listNunique[i], listMFunique[j]]
            point_4 = [listNunique[i], listMFunique[j + 1]]
            point_2 = [listNunique[i + 1], listMFunique_plusone[j]]
            point_3 = [listNunique[i + 1], listMFunique_plusone[j + 1]]

            test = is_inside_quadrilateral(
                (x, y), np.array([point_1, point_2, point_3, point_4]), margin=0.0001
            )
            if test:
                coords = [point_1, point_2, point_3, point_4]

                return coords

    return False


def interp(df, x_new, y_new, name_mf="Mass Flow", return_quad=False):
    """
    Get the coordinate of the the quadrilateral in which (x,y) is

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        x_new : float, particular N
        y_new : float, particular second values (Mass Flow most of the time)
        name_mf : str, name of the second feature (first one is always N and second one is Mass Flow by default)


    Returns:
        z_interp1 : float, the interpolated value for Efficiency
        z_interp2 : float, the interpolated value for Pressure Ratio
        * Return np.nan if no quadrilateral was found to interpolate
    """

    coords = search_point(
        x_new, y_new, name_mf, df
    )  # Get coordinate of quadrilateral in which we can find (x_new, y_new)

    z_interp1, z_interp2 = np.nan, np.nan

    if not coords:  # If no quadrilateral was found then outputs np.nan
        # print('pb coords')
        return z_interp1, z_interp2

    z = np.zeros(4)

    for i in range(4):  # For each coordinate get the maximum Efficiency
        index = (df["N"] == coords[i][0]) & (df[name_mf] == coords[i][1])
        z[i] = np.max(df.loc[index, "Efficiency"])

    try:  # Interpolate Efficiency value for (x_new, y_new) from coords and z
        z_interp1 = griddata(coords, z, (x_new, y_new), method="linear")
    except:  # Otherwise just take the mean
        z_interp1 = np.mean(z)

    if np.isnan(z_interp1):
        try:
            z_interp1 = griddata(coords, z, (x_new, y_new), method="nearest")
        except:
            z_interp1 = np.mean(z)

    # Same as earlier but with Pressure Ratio

    z = np.zeros(4)

    for i in range(4):
        index = (df["N"] == coords[i][0]) & (df[name_mf] == coords[i][1])
        z[i] = np.max(df.loc[index, "Pressure Ratio"])

    try:
        z_interp2 = griddata(coords, z, (x_new, y_new), method="linear")
    except:
        z_interp2 = np.mean(z)

    if np.isnan(z_interp2):
        try:
            z_interp2 = griddata(coords, z, (x_new, y_new), method="nearest")
        except:
            z_interp2 = np.mean(z)

    #    except:

    #        print('x,y, other pb ?', x_new, y_new)

    if return_quad:
        return z_interp1, z_interp2, coords
    else:
        return z_interp1, z_interp2


def get_interp_SRA(df, inputs, name_mf="Mass Flow"):
    """
    Interpolate values for Efficiency and Pressure Ratio from N and Mass Flow

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        inputs : List of size 2 of inputs, the new N (with degradation) and the observed Mass Flow
        name_mf : str, name of the second feature (first one is always N and second one is Mass Flow by default)


    Returns:
        output1 : float, the interpolated value for Efficiency
        output2 : float, the interpolated value for Pressure Ratio
        * Return np.nan if interp functions did not find interpolation
    """

    output1, output2 = interp(df, inputs[0], inputs[1], name_mf)

    return output1, output2


def interp_beta(data, input1, input2, input_names, output_names):
    
    data = data.sort_values(input_names)
    fit_points = list(zip(data[input_names[0]], data[input_names[1]]))

    if len(output_names) == 2:  # If output_names contains 2 names
        values = list(zip(data[output_names[0]], data[output_names[1]]))
    elif len(output_names) == 1:
        values = data[output_names[0]]
    elif len(output_names) == 3:  # If output_names contains 2 names
        values = list(zip(data[output_names[0]], data[output_names[1]], data[output_names[2]]))

    interp = LinearNDInterpolator(fit_points, values)
    Z = interp(input1, input2)

    return Z


def get_interp_beta(df, inputs, input_names, output_names):
    """
    Interpolate values for Efficiency, Pressure Ratio and Mass Flow from N and beta

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        inputs : List of size 2 of inputs, the new N (with degradation) and the observed Mass Flow


    Returns:
        output1 : float, the interpolated value for Efficiency
        output2 : float, the interpolated value for Pressure Ratio
        output3 : float, the interpolated value for Mass Flow
        * Return np.nan if interp functions did not find interpolation
    """
    output1, output2, output3 = interp_beta(df, inputs[0], inputs[1], input_names, output_names)

    return output1, output2, output3    



##########################
#     Operating Line     #
##########################


def add_data_degrad_beta(df, N, degrad_comp=1):
    """
    Compute the shifted compressor map associated with the degradation input

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular N without any degradation (point of reference in df)
        degrad_comp : float ]0,1.1], coefficient of degradation of the compressor map



    Returns:
        data : Pandas DataFrame, dataframe with the data for the compressor map associated with N*degrad_comp added to the input dataframe df
    """

    if degrad_comp == 1:
        return df  # If no degradation, then just return data
    else:  # Else compute the shift of the compressor map for N and degrad_comp
        data_n = df[df.N == N]

        # Compute degraded values of the features
        new_beta = data_n["beta"]
        new_N = data_n["N"] * degrad_comp
        new_Mass_flow = data_n["Mass Flow"] * degrad_comp
        new_Pressure_Ratio = data_n["Pressure Ratio"] * degrad_comp
        new_Efficiency = data_n["Efficiency"] * degrad_comp
        new_Mass_flow_Lbs = data_n["Mass Flow Lbs"] * degrad_comp

        # Create new dataframe with degraded values
        new_data = pd.DataFrame(
            {
                "beta": new_beta,
                "N": new_N,
                "Mass Flow": new_Mass_flow,
                "Pressure Ratio": new_Pressure_Ratio,
                "Efficiency": new_Efficiency,
                "Mass Flow Lbs": new_Mass_flow_Lbs,
            }
        )

        # Concatenate df with new values from new_data
        data = pd.concat([df, new_data])
        data = data.sort_values(["N", "Mass Flow"])
        data = data.reset_index(drop=True)

        return data

def add_data_degrad(df, N, degrad_comp=1):
    """
    Compute the shifted compressor map associated with the degradation input

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular N without any degradation (point of reference in df)
        degrad_comp : float ]0,1.1], coefficient of degradation of the compressor map



    Returns:
        data : Pandas DataFrame, dataframe with the data for the compressor map associated with N*degrad_comp added to the input dataframe df
    """

    if degrad_comp == 1:
        return df  # If no degradation, then just return data
    else:  # Else compute the shift of the compressor map for N and degrad_comp
        data_n = df[df.N == N]

        # Compute degraded values of the features
        new_N = data_n["N"] * degrad_comp
        new_Mass_flow = data_n["Mass Flow"] * degrad_comp
        new_Pressure_Ratio = data_n["Pressure Ratio"] * degrad_comp
        new_Efficiency = data_n["Efficiency"] * degrad_comp
        new_Mass_flow_Lbs = data_n["Mass Flow Lbs"] * degrad_comp

        # Create new dataframe with degraded values
        new_data = pd.DataFrame(
            {
                "N": new_N,
                "Mass Flow": new_Mass_flow,
                "Pressure Ratio": new_Pressure_Ratio,
                "Efficiency": new_Efficiency,
                "Mass Flow Lbs": new_Mass_flow_Lbs,
            }
        )

        # Concatenate df with new values from new_data
        data = pd.concat([df, new_data])
        data = data.sort_values(["N", "Mass Flow"])
        data = data.reset_index(drop=True)

        return data
##########################
#     Operating Line     #
##########################


def get_operating_line(df, W3R):
    """
    Compute the operating line of the engine based on W3R expected values and the map compressor

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        W3R : float, expected mass air flow at the end of the compressor

    Returns:
        ope_line : Pandas DataFrame, dataframe with the data corresponding to the operating line of the engine
    """
    mass_flow_out, OPR_out, eff_out = [], [], []
    data = get_possible_outflow(df)

    for n in data.N.unique():
        data_n = data[data.N == n]
        data_n = data_n.sort_values("W3R")
        mass_flow_op = np.interp(W3R, data_n["W3R"], data_n["Mass Flow"])
        mass_flow_out.append(mass_flow_op)

        data_n = data_n.sort_values("Mass Flow")
        OPR_op = np.interp(mass_flow_op, data_n["Mass Flow"], data_n["Pressure Ratio"])
        eff_op = np.interp(mass_flow_op, data_n["Mass Flow"], data_n["Efficiency"])

        OPR_out.append(OPR_op)
        eff_out.append(eff_op)

    ope_line = pd.DataFrame(
        {
            "N": data.N.unique(),
            "Mass Flow": mass_flow_out,
            "Pressure Ratio": OPR_out,
            "Efficiency": eff_out,
        }
    )

    ## We could also get corresponding OPR and eff in this function but not sure it's useful

    return ope_line


def get_operation_point(df, N, degrad_comp, W3R=4, P2=191801.04700911307, T2=360.0):
    """
    Compute the operating point of the engine based on W3R expected values, the map compressor and the particular N and degradation index

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular Engine Power for which we want to compute the operation point
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map
        W3R : float, expected mass air flow at the end of the compressor

    Returns:
        opti_mass_flow : float, the Mass air flow corresponding to the operation point for the choosen engine power
    """

    data_n = get_possible_outflow_degrad(df, N, degrad_comp, P2, T2)
    data_n = data_n.sort_values("Degraded Mass Flow")
    opti_mass_flow = np.interp(W3R, data_n["W3R"], data_n["Degraded Mass Flow"])

    return opti_mass_flow


def create_data_with_newN(df, N, return_quad=False):
    """
    Create dataframe for an engine power that is not in the original data for the compressor map. That is values for N outside of
    [0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1, 1.025]

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular Engine Power for which we want new data

    Returns:
        data_new : Pandas DataFrame, dataframe with the data corresponding to the N input
    """
    ## First commpute min and max of flow for corresponding N
    max_par_groupe = df.groupby("N")["Mass Flow"].max().reset_index()
    min_par_groupe = df.groupby("N")["Mass Flow"].min().reset_index()

    # Then interpolate values for the min and max Air Flow
    min_flow = np.interp(N, min_par_groupe["N"], min_par_groupe["Mass Flow"])
    max_flow = np.interp(N, max_par_groupe["N"], max_par_groupe["Mass Flow"])

    # Compute as many points between min and max as in the original dataframe for a particular N
    mass_flow_new = np.linspace(min_flow, max_flow, num=len(df[df.N == 0.9]))

    # Interpolate values for each mass_flow
    values_pr = np.array([interp(df, N, m)[1] for m in mass_flow_new])
    values_eff = np.array([interp(df, N, m)[0] for m in mass_flow_new])
    if return_quad:
        coords_quad = np.array(
            [interp(df, N, m, return_quad=True)[2] for m in mass_flow_new]
        )
    N_new = [N] * len(values_pr)

    # Create new dataframe
    data_new = pd.DataFrame(
        {
            "N": N_new,
            "Mass Flow": mass_flow_new,
            "Efficiency": values_eff,
            "Pressure Ratio": values_pr,
        }
    )
    data_new["Mass Flow Lbs"] = 2.2046226218488 * data_new["Mass Flow"]
    if return_quad:
        return data_new, coords_quad

    else:
        return data_new

def create_data_with_newN_beta(data, N):
    beta = data.beta.unique()
    interpolated_values = interp_beta(data, 0.95, beta, ["N", "beta"], ["Efficiency", "Pressure Ratio", "Mass Flow"])

    data_out = pd.DataFrame(interpolated_values)
    data_out.columns = ["Efficiency", "Pressure Ratio", "Mass Flow"]

    data_out["beta"] = beta
    data_out["N"] = [N]*len(beta)
    data_out["Mass Flow Lbs"] = 2.2046226218488 * data_out["Mass Flow"]

    return data_out


def create_data(file, N, degrad_comp):
    """
    Create dataframe for an engine power that is not in the original data for the compressor map. That is values for N outside of
    [0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1, 1.025]. And add shifted map from the degradation degrad_comp

    Parameters:
        file : str, string with the path to get the map compressor data
        N : float, particular Engine Power for which we want new data
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map

    Returns:
        data : Pandas DataFrame, dataframe with the data corresponding to the N input and degrad_comp along with original data
    """
    data = parse_file(file)
    if N not in data["N"].unique():
        new_data = create_data_with_newN(data, N)
        data = pd.concat([data, new_data])
        data = data.sort_values(["N", "Mass Flow"])
        data = data.reset_index(drop=True)
    data = add_data_degrad(data, N, degrad_comp)

    return data

def create_data_beta(file, N, degrad_comp):
    """
    Create dataframe for an engine power that is not in the original data for the compressor map. That is values for N outside of
    [0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1, 1.025]. And add shifted map from the degradation degrad_comp

    Parameters:
        file : str, string with the path to get the map compressor data
        N : float, particular Engine Power for which we want new data
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map

    Returns:
        data : Pandas DataFrame, dataframe with the data corresponding to the N input and degrad_comp along with original data
    """
    data = parse_file_beta(file)
    if N not in data["N"].unique():
        new_data = create_data_with_newN_beta(data, N)
        data = pd.concat([data, new_data])
        data = data.sort_values(["N", "beta"])
        data = data.reset_index(drop=True)
    data = add_data_degrad_beta(data, N, degrad_comp)

    return data

def create_data_from_csv(csv_file, N, degrad_comp):
    """
    Create dataframe for an engine power that is not in the original data for the compressor map. That is values for N outside of
    [0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1, 1.025]. And add shifted map from the degradation degrad_comp

    Parameters:
        file : str, string with the path to get the map compressor data
        N : float, particular Engine Power for which we want new data
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map

    Returns:
        data : Pandas DataFrame, dataframe with the data corresponding to the N input and degrad_comp along with original data
    """
    data = pd.read_csv(csv_file)
    if N not in data["N"].unique():
        new_data = create_data_with_newN(data, N)
        data = pd.concat([data, new_data])
        data = data.sort_values(["N", "Mass Flow"])
        data = data.reset_index(drop=True)
    data = add_data_degrad(data, N, degrad_comp)

    return data


def get_possible_outflow(df, P2=191801.04700911307, T2=360.0):
    """
    Compute the mass flow exiting the compressor map for each value of W2R and add it to the original dataframe

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        P2 : float, Air Pressure before entering the compressor
        T2 : float, Air Temperature before entering the compressor

    Returns:
        data_out : Pandas DataFrame, dataframe with new columns W3R along with original data
    """

    gamma = 1.4
    data_out = df.copy()
    ### Get all possible PR for all possible W2R, get the corresponding efficiency. From this compute P3 and T3 and then a min and max for W3R
    data_out["P3"] = P2 * data_out["Pressure Ratio"]
    data_out["T3"] = T2 * (
        1
        + (data_out["Pressure Ratio"] ** ((gamma - 1) / gamma) - 1)
        / data_out["Efficiency"]
    )

    data_out["W3R"] = data_out["Mass Flow"] * (
        np.sqrt(data_out["T3"]) * P2 / (data_out["P3"] * np.sqrt(T2))
    )
    return data_out


def get_possible_outflow_degrad(df, N, degrad_comp, P2=191801.04700911307, T2=360.0):
    """
    Compute the mass flow exiting the compressor map for particular N and degrad_comp for each value of W2R and add it to the original dataframe

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular Engine Power for which we want W3R
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map
        P2 : float, Air Pressure before entering the compressor
        T2 : float, Air Temperature before entering the compressor

    Returns:
        data_out : Pandas DataFrame, dataframe with new columns W3R along with original data
    """
    data_out = df[df.N == N]
    if len(data_out) == 0:
        data_out = create_data_with_newN(df, N)

    data_out["Degraded Pressure Ratio"] = degrad_comp * data_out["Pressure Ratio"]
    data_out["Degraded Efficiency"] = degrad_comp * data_out["Efficiency"]
    data_out["Degraded Mass Flow"] = degrad_comp * data_out["Mass Flow"]

    gamma = 1.4
    ### Get all possible PR for all possible W2R, get the corresponding efficiency. From this compute P3 and T3 and then a min and max for W3R
    data_out["P3"] = P2 * data_out["Degraded Pressure Ratio"]
    data_out["T3"] = T2 * (
        1
        + (data_out["Degraded Pressure Ratio"] ** ((gamma - 1) / gamma) - 1)
        / data_out["Degraded Efficiency"]
    )

    data_out["W3R"] = data_out["Degraded Mass Flow"] * (
        np.sqrt(data_out["T3"]) * P2 / (data_out["P3"] * np.sqrt(T2))
    )

    return data_out


def get_minmax(df, N, degrad_comp, P2, T2, Tref=288.15, Pref=101325, verbose=0):
    """
    Compute minimum and maximum Mass Air Flow that enter the engine for given N and degrad_comp

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular Engine Power for which we want W3R
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map
        P2 : float, Air Pressure before entering the compressor
        T2 : float, Air Temperature before entering the compressor
        Tref : float, Air Temperature reference for reduction of the features
        Pref : float, Air Pressure reference for reduction of the features
        verbose : do you want to print information in case of error

    Returns:
        min_W0 : float, minimum Air Flow entering the engine at plan 0
        max_W0 : float, maximum Air Flow entering the engine at plan 0
    """

    data_N = df[df.N == N]
    data_N["Degraded Mass Flow"] = degrad_comp * data_N["Mass Flow"]
    min_W0, max_W0 = (
        np.min(data_N["Degraded Mass Flow"]),
        np.max(data_N["Degraded Mass Flow"]),
    )
    if verbose:
        print("Min W0 : ", min_W0, ", Max W0 : ", max_W0)
        print(
            "Min W2R : ",
            min_W0 * (P2 / Pref) / np.sqrt(T2 / Tref),
            ", Max W2R : ",
            max_W0 * (P2 / Pref) / np.sqrt(T2 / Tref),
        )

    return min_W0, max_W0


##########################
#         Plot           #
##########################


def plot_interp(df, name_plot="Efficiency", name_mf="Mass Flow"):
    """
    Plot interpolation results

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        name_plot : str, name of the feature to interpolate and plot
        name_mf : str, name of the input feature for interpolation

    Returns:
        None
        Plot interpolated name_plot vs name_mf
    """

    plt.figure(1, figsize=(15, 15))
    # Get the max name_mf for each value of N
    max_par_groupe = df.groupby("N")[name_mf].max().reset_index()
    # Get the min name_mf for each value of N
    min_par_groupe = df.groupby("N")[name_mf].min().reset_index()

    for N in df["N"].unique():
        index = df["N"] == N
        plt.plot(df.loc[index, name_mf], df.loc[index, name_plot], "--*")

    dn = df["N"].unique()

    for n1, n2 in zip(dn[0:-1], dn[1:]):
        n = (n1 + n2) / 2

        min_flow = np.interp(n, min_par_groupe["N"], min_par_groupe[name_mf])
        max_flow = np.interp(n, max_par_groupe["N"], max_par_groupe[name_mf])

        delta = (max_flow - min_flow) / 25

        mass_flow = np.arange(min_flow, max_flow + delta, delta)

        # print("n", n, "m", m)

        # interp(n, m)

        plt.figure(2)

        values = np.array([interp(n, m)[0] for m in mass_flow])

        plt.figure(1)

        plt.plot(mass_flow, values, "*-")


def plot_W0_evolution(df, N, list_W0, degrad_comp=1, save=False, path_save=None):
    """
    Plot W0 evolution during the iteration in the simulator function of the engine class

    Parameters:
        df : Pandas DataFrame, dataframe with the data
        N : float, particular Engine Power for which we computed list of W0
        list_W0 : List of float, List with the values of W0 as they are computed along iteration of the engine until found the correct entering Mass Air Flow
        degrad_comp : float, ]0,1.1], coefficient of degradation of the compressor map

    Returns:
        None
        Plot Pressure Ratio and Efficiency versus Mass Flow and add evolution of W0 with red arrows
    """
    data_n = df[df.N == N]

    data_n["Degraded Pressure Ratio"] = degrad_comp * data_n["Pressure Ratio"]
    data_n["Degraded Efficiency"] = degrad_comp * data_n["Efficiency"]
    data_n["Degraded Mass Flow"] = degrad_comp * data_n["Mass Flow"]

    list_pressure_ratio, list_eff = [], []
    data_n = data_n.sort_values("Degraded Mass Flow")
    for wo in list_W0:
        temp_pressure_ratio = np.interp(
            wo, data_n["Degraded Mass Flow"], data_n["Degraded Pressure Ratio"]
        )

        list_pressure_ratio.append(temp_pressure_ratio)
        temp_eff = np.interp(
            wo, data_n["Degraded Mass Flow"], data_n["Degraded Efficiency"]
        )
        list_eff.append(temp_eff)

    ## First let us plot just for Pressure ratio
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_n["Degraded Mass Flow"],
            y=data_n["Degraded Pressure Ratio"],
            mode="lines",
            name="Pressure Ratio",
        )
    )

    fig.add_trace(
        go.Scatter(x=list_W0, y=list_pressure_ratio, mode="markers", name="Iterated W0")
    )

    fig.add_trace(
        go.Scatter(
            x=list_W0,
            y=list_pressure_ratio,
            mode="markers+lines",
            marker=dict(
                size=10, symbol="arrow-bar-up", angleref="previous", color="red"
            ),
            showlegend=False,
        )
    )

    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_n["Degraded Mass Flow"],
            y=data_n["Degraded Efficiency"],
            mode="lines",
            name="Efficiency",
        )
    )

    fig.add_trace(go.Scatter(x=list_W0, y=list_eff, mode="markers", name="Iterated W0"))

    fig.add_trace(
        go.Scatter(
            x=list_W0,
            y=list_eff,
            mode="markers+lines",
            marker=dict(
                size=10, symbol="arrow-bar-up", angleref="previous", color="red"
            ),
            showlegend=False,
        )
    )

    fig.show()


##########################
#     Optimization       #
##########################


def derivative_function(f, x, h=0.0001):
    """
    Compute the value of the derivative of f at point x

    Parameters:
        f : function, the function that we need to compute the derivative for
        x : float, the point at which we want to compute the derivative
        h : float, h value for computing the derivative according to the formula f'(x0) = (f(x0) - f(x0+h))/h

    Returns:
        y : float, y = f'(x)
    """
    y = (f(x + h) - f(x)) / h
    return y


def newton(f, Df, x0, epsilon, max_iter, P2, T2, alpha=1):
    """Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.
    alpha : float
        scale factor for the update step. Lower alpha means smaller jumps between xn+1 and xn

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10,1)
    Found solution after 5 iterations.
    1.618033988749989
    """
    Pref = 101325
    Tref = 288.15

    list_xn = [x0 * np.sqrt(T2 / Tref) / (P2 / Pref)]
    xn = x0
    count = 0
    for n in range(0, max_iter):
        count += 1
        fxn = f(xn)
        if abs(fxn) < epsilon:
            # print('Found solution after',n,'iterations.')
            return xn, list_xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return xn, list_xn

        xn = xn - fxn * alpha / Dfxn
        list_xn.append(xn * np.sqrt(T2 / Tref) / (P2 / Pref))
        # print("Current x ", xn, "\n")

    print("Exceeded maximum iterations. No solution found.")
    return xn, list_xn
