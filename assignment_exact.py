import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp

def solve_ptl_problem(file_path):
    # Load data
    df_positions_zones = pd.read_excel(file_path, sheet_name='Tiempo_salida', index_col=0)
    df_orders_skus = pd.read_excel(file_path, sheet_name='Tiempo_SKU', index_col=0)
    df_workers = pd.read_excel(file_path, sheet_name='Productividad')
    try:
        df_parameters = pd.read_excel(file_path, sheet_name="Parametros")
    except ValueError:
        df_parameters = pd.DataFrame({"v": [3.0]})
        print("Advertencia: No se encontró la hoja 'Parametros', se usará un valor por defecto")
    
    # Extract sets
    orders = list(df_orders_skus.index)
    zones = list(df_positions_zones.index)
    positions = list(df_positions_zones.columns)
    skus = list(df_orders_skus.columns)
    workers = list(df_workers['Trabajadores'].values) if 'Trabajadores' in df_workers.columns else []
    
    # Calculate parameters
    positions_in_each_zone = (df_positions_zones > 0)
    cnt_positions_per_zone = positions_in_each_zone.sum(axis=1)
    sku_belongs_to_order = (df_orders_skus > 0)
    v = df_parameters['v'].values[0]
    
    # Create position to zone mapping
    position_zone = {}
    for zone, row in positions_in_each_zone.iterrows():
        zone_positions = positions_in_each_zone.columns[row.values]
        for p in zone_positions:
            position_zone[p] = zone
    
    # Pre-calculate times
    order_processing_time = df_orders_skus.sum(axis=1)
    travel_time = 2 * (df_positions_zones / v)
    travel_time_per_position = travel_time.sum(axis=0)
    
    # Create the solver
    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver('CBC')
    
    if not solver:
        return None
    
    # Create variables
    # X_ik: Binary variable that takes the value of 1 if order i is assigned to position k
    x = {}
    for i in orders:
        for k in positions:
            x[i, k] = solver.BoolVar(f'x_{i}_{k}')
    
    # W_j: Total processing time in zone j
    w = {}
    for j in zones:
        w[j] = solver.NumVar(0, solver.infinity(), f'w_{j}')
    
    # Create W_max and W_min variables for balancing
    w_max = solver.NumVar(0, solver.infinity(), 'w_max')
    w_min = solver.NumVar(0, solver.infinity(), 'w_min')
    
    # Constraints
    
    # Constraint 1: Each order must be assigned to exactly one position
    for i in orders:
        solver.Add(sum(x[i, k] for k in positions) == 1)
    
    # Constraint 2: Each position can have at most one order assigned
    for k in positions:
        solver.Add(sum(x[i, k] for i in orders) <= 1)
    
    # Constraint 3: The orders assigned to a zone cannot exceed the total positions of that zone
    for j in zones:
        zone_positions = [k for k in positions if position_zone.get(k) == j]
        solver.Add(sum(x[i, k] for i in orders for k in zone_positions) <= cnt_positions_per_zone[j])
    
    # Constraint 4: Calculate the total processing time in each zone
    for j in zones:
        zone_positions = [k for k in positions if position_zone.get(k) == j]
        solver.Add(w[j] == sum(x[i, k] * (order_processing_time[i] + travel_time_per_position[k]) 
                              for i in orders for k in zone_positions))
    
    # Constraint 5: Define W_max and W_min
    for j in zones:
        solver.Add(w[j] <= w_max)
        solver.Add(w[j] >= w_min)
    
    # Objective: Minimize the difference between max and min processing times
    solver.Minimize(w_max)
    
    # Solve the problem, show verbose output
    # solver.EnableOutput()
    solver.SetTimeLimit(10*6*1000)  # 10 minute
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # Extract the solution
        assignments = []
        for i in orders:
            for k in positions:
                if x[i, k].solution_value() > 0.5:  # checking if the binary variable is 1
                    cost = order_processing_time[i] + travel_time_per_position[k]
                    assignments.append((i, k, cost))
        
        # Calculate zone costs
        zone_cost = {j: 0 for j in zones}
        for order, position, cost in assignments:
            zone_cost[position_zone[position]] += cost
        
        return {
            'status': 'optimal',
            'assignments': sorted(assignments, key=lambda x: int(x[0].split('_')[1])),
            'zone_cost': zone_cost,
            'objective_value': solver.Objective().Value(),
            'w_max': w_max.solution_value(),
            'w_min': w_min.solution_value()
        }
    else:
        return {
            'status': 'not optimal: ' + str(solver.status),
            'objective_value': None
        }

def display_results(result):
    if result['status'] == 'optimal':
        print("Optimal solution found!")
        print(f"Objective value (max-min difference): {result['objective_value']}")
        print(f"Maximum zone time: {result['w_max']}")
        print(f"Minimum zone time: {result['w_min']}")
        print("\nAssignments (Order -> Position):")
        for order, position, cost in result['assignments']:
            print(f"{order} -> {position} (Cost: {cost:.2f})")
        print("\nZone Costs:")
        for zone, cost in result['zone_cost'].items():
            print(f"Zone {zone}: {cost:.2f}")
    else:
        print("No optimal solution found:", result['status'])

if __name__ == "__main__":
    tasks = [
        (
            "data/Data_PTL/Data_40_Salidas_composición_zonas_homo.xlsx",
            "Resultados_PTL/res_40_comp_homo.xlsx",
        ),
        (
            "data/Data_PTL/Data_40_Salidas_composición_zonas_hetero.xlsx",
            "Resultados_PTL/res_40_comp_hetero.xlsx",
        ),
        (
            "data/Data_PTL/Data_60_Salidas_composición_zonas_hetero.xlsx",
            "Resultados_PTL/res_60_comp_hetero.xlsx",
        ),
        (
            "data/Data_PTL/Data_60_Salidas_composición_zonas_homo.xlsx",
            "Resultados_PTL/res_60_comp_homo.xlsx",
        ),
        (
            "data/Data_PTL/Data_80_Salidas_composición_zonas_hetero.xlsx",
            "Resultados_PTL/res_80_comp_hetero.xlsx",
        ),
        (
            "data/Data_PTL/Data_80_Salidas_composición_zonas_homo.xlsx",
            "Resultados_PTL/res_80_comp_homo.xlsx",
        ),
    ]
    for file_path, output_file in tasks:
        print(file_path)
        result = solve_ptl_problem(file_path)
        display_results(result)