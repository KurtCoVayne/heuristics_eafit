import pandas as pd
import numpy as np


def cargar_datos(file_path):
    """
    Carga los datos desde el archivo Excel.

    Args:
        file_path: Ruta al archivo Excel con los datos del problema

    Returns:
        Tupla con los dataframes y parámetros cargados
    """
    df_positions_zones = pd.read_excel(
        file_path, sheet_name="Tiempo_salida", index_col=0
    )
    df_orders_skus = pd.read_excel(file_path, sheet_name="Tiempo_SKU", index_col=0)
    df_workers = pd.read_excel(file_path, sheet_name="Productividad")

    try:
        df_parameters = pd.read_excel(file_path, sheet_name="Parametros")
    except ValueError:
        df_parameters = pd.DataFrame({"v": [3.0]})
        print("Advertencia: No se encontró la hoja 'Parametros', se usará un valor por defecto")


    return df_positions_zones, df_orders_skus, df_workers, df_parameters


def preparar_datos(df_positions_zones, df_orders_skus, df_parameters):
    """
    Prepara los datos para resolver el problema, extrayendo conjuntos y parámetros.

    Args:
        df_positions_zones: DataFrame con la relación entre posiciones y zonas
        df_orders_skus: DataFrame con la relación entre órdenes y SKUs
        df_parameters: DataFrame con los parámetros generales

    Returns:
        Tupla con los conjuntos y parámetros calculados
    """
    # Corresponde a la carga de los conjuntos P (órdenes), Z (zonas), S (posiciones), R (SKUs)
    orders = list(df_orders_skus.index)
    zones = list(df_positions_zones.index)
    positions = list(df_positions_zones.columns)
    skus = list(df_orders_skus.columns)

    # Cálculo de parámetros s_jk (positions_in_each_zone): Parámetro binario que indica
    # si la posición k \in S es parte de la zona j \in Z, entonces existe un tiempo de salida mayor a 0, sino 0
    positions_in_each_zone = df_positions_zones > 0
    # Parámetro ns_j (cnt_positions_per_zone): Número de posiciones en cada zona j, si existe un tiempo de salida mayor a 0
    cnt_positions_per_zone = positions_in_each_zone.sum(axis=1)
    # Parámetro rp_im (sku_belongs_to_order): Parámetro binario que indica, no se usa
    # sku_belongs_to_order = (df_orders_skus > 0)
    _ = df_orders_skus > 0
    # Parámetro v: Velocidad promedio de los trabajadores
    v = df_parameters["v"].values[0]

    # Crear mapeo de posición a zona
    position_zone = {}
    zone_positions = {}
    for zone, row in positions_in_each_zone.iterrows():
        zone_positions[zone] = positions_in_each_zone.columns[row.values].tolist()
        for p in zone_positions[zone]:
            position_zone[p] = zone

    # Cálculo de tiempos de procesamiento según las expresiones (7) y (8) del modelo
    # order_processing_time: Corresponde a tr_im (tiempo para clasificar un SKU) y solo depende de la orden
    order_processing_time = df_orders_skus.sum(axis=1)

    # travel_time: Corresponde a d_jk/v (distancia/velocidad) según la expresión (7) y solo depende de la posición
    travel_time = 2 * (df_positions_zones / v)
    travel_time_per_position = travel_time.sum(axis=0)

    # Cálculo del costo total por orden y posición según la expresión (7)
    # Y_i = sum((tr_im + 2(d_jk/v)) * rp_im * X_ik * s_jk)
    order_position_cost = {}
    for i in orders:
        for k in positions:
            order_position_cost[(i, k)] = (
                order_processing_time[i] + travel_time_per_position[k]
            )

    return (
        orders,
        zones,
        positions,
        skus,
        cnt_positions_per_zone,
        position_zone,
        zone_positions,
        order_processing_time,
        travel_time_per_position,
        order_position_cost,
    )


def asignar_ordenes(
    orders, zones, positions, position_zone, order_position_cost, cnt_positions_per_zone
):
    """
    Asigna órdenes a posiciones utilizando una heurística constructiva voraz.
    """
    # Implementa una heurística para minimizar el tiempo máximo entre zonas
    # que corresponde a la función objetivo (3): Minimizar W_max

    # Ordena las órdenes de mayor a menor tiempo de procesamiento
    sorted_orders = sorted(
        orders,
        key=lambda i: sum([order_position_cost[(i, k)] for k in positions])
        / len(positions),
        reverse=True,
    )

    # Inicializa variables para seguimiento de asignaciones
    assignments = []  # Lista de asignaciones (orden, posición, costo)
    zone_load = {
        j: 0 for j in zones
    }  # W_j: Tiempo total por zona según restricción (8)
    position_assigned = {
        k: False for k in positions
    }  # Seguimiento de posiciones asignadas
    zone_remaining_positions = {
        j: cnt_positions_per_zone[j] for j in zones
    }  # Posiciones disponibles

    # Asignación voraz - Implementa la restricción (4): Cada orden se asigna a una posición
    # y la restricción (5): Cada posición tiene máximo una orden asignada
    for i in sorted_orders:
        best_position = None
        min_max_zone_time = float("inf")

        for k in positions:
            if position_assigned[k]:
                continue

            current_zone = position_zone[k]

            # Restricción (6): No asignar más órdenes que posiciones en la zona
            if zone_remaining_positions[current_zone] <= 0:
                continue

            # Calcula nuevo W_j si asignamos esta orden a esta posición
            new_zone_load = zone_load[current_zone] + order_position_cost[(i, k)]

            # Busca minimizar W_max según objetivo (3)
            temp_zone_load = zone_load.copy()
            temp_zone_load[current_zone] = new_zone_load
            potential_max_zone_time = max(temp_zone_load.values())

            if potential_max_zone_time < min_max_zone_time:
                min_max_zone_time = potential_max_zone_time
                best_position = k

        # Realiza la asignación
        if best_position is not None:
            current_zone = position_zone[best_position]
            zone_load[current_zone] += order_position_cost[(i, best_position)]
            position_assigned[best_position] = True
            zone_remaining_positions[current_zone] -= 1
            assignments.append(
                (i, best_position, order_position_cost[(i, best_position)])
            )
        else:
            print(f"Advertencia: No se pudo asignar la orden {i}")

    return assignments, zone_load


def evaluar_solucion(assignments, zone_load):
    """
    Evalúa la solución obtenida calculando métricas clave.

    Args:
        assignments: Lista de asignaciones (orden, posición, costo)
        zone_load: Diccionario con la carga de cada zona

    Returns:
        Diccionario con los resultados de la evaluación
    """
    # Calcular valor objetivo y tiempo mínimo de zona
    w_max = max(zone_load.values())
    w_min = min(zone_load.values())

    return {
        "status": "solución heurística",
        "assignments": sorted(
            assignments, key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else x[0]
        ),
        "zone_cost": zone_load,
        "objective_value": w_max,
        "w_max": w_max,
        "w_min": w_min,
    }


def generar_excel_asignaciones(result, output_file):
    """
    Genera un archivo Excel con las asignaciones de órdenes a posiciones.

    Args:
        result: Diccionario con los resultados de la solución
        output_file: Ruta del archivo Excel a generar
    """
    # Crear un DataFrame con las asignaciones
    df_assignments = pd.DataFrame(
        [(order, position) for order, position, _ in result["assignments"]],
        columns=["Pedido", "Salida"],
    )

    # Crear un DataFrame con los costos por zona
    df_zone_costs = pd.DataFrame(
        [(zone, cost) for zone, cost in result["zone_cost"].items()],
        columns=["Zona", "Costo Total"],
    )

    # Crear un DataFrame con las métricas
    df_metrics = pd.DataFrame(
        [
            ["Tiempo máximo de zona (W_max)", result["w_max"]],
            ["Tiempo mínimo de zona (W_min)", result["w_min"]],
            ["Diferencia (W_max - W_min)", result["w_max"] - result["w_min"]],
        ],
        columns=["Métrica", "Valor"],
    )

    # Crear un escritor de Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_assignments.to_excel(writer, sheet_name="Asignaciones", index=False)
        df_zone_costs.to_excel(writer, sheet_name="Costos por Zona", index=False)
        df_metrics.to_excel(writer, sheet_name="Métricas", index=False)

    print(f"Archivo Excel generado: {output_file}")


def mostrar_resultados(result, position_zone):
    """
    Muestra los resultados de la solución en la consola.

    Args:
        result: Diccionario con los resultados de la solución
    """
    print(f"¡Solución encontrada utilizando enfoque heurístico!")
    print(f"Valor objetivo (tiempo máximo de zona): {result['objective_value']:.2f}")
    print(f"Tiempo máximo de zona: {result['w_max']:.2f}")
    print(f"Tiempo mínimo de zona: {result['w_min']:.2f}")
    print(f"Diferencia máx-mín: {result['w_max'] - result['w_min']:.2f}")
    print("\nAsignaciones (Pedido -> Posición):")
    for order, position, cost in result["assignments"]:
        print(
            f"{order} -> {position} (Costo: {cost:.2f}, Zona: {position_zone[position]})"
        )
    print("\nCostos por Zona:")
    for zone, cost in sorted(result["zone_cost"].items()):
        print(f"Zona {zone}: {cost:.2f}")


def solve_ptl_heuristic(file_path):
    """
    Función principal que resuelve el problema PTL utilizando una heurística constructiva.

    Args:
        file_path: Ruta al archivo Excel con los datos del problema

    Returns:
        Diccionario con los resultados de la solución
    """
    # 1. Cargar datos
    df_positions_zones, df_orders_skus, _, df_parameters = cargar_datos(file_path)

    # 2. Preparar datos
    (
        orders,
        zones,
        positions,
        _,
        cnt_positions_per_zone,
        position_zone,
        _,
        _,
        _,
        order_position_cost,
    ) = preparar_datos(df_positions_zones, df_orders_skus, df_parameters)

    # 3. Asignar órdenes a posiciones
    assignments, zone_load = asignar_ordenes(
        orders,
        zones,
        positions,
        position_zone,
        order_position_cost,
        cnt_positions_per_zone,
    )

    # 4. Evaluar la solución
    result = evaluar_solucion(assignments, zone_load)

    return result, position_zone


# Ejemplo de uso:
if __name__ == "__main__":

    tasks = [
        (
            "Data_PTL/Data_40_Salidas_composición_zonas_homo.xlsx",
            "Resultados_PTL/res_40_comp_homo.xlsx",
        ),
        (
            "Data_PTL/Data_40_Salidas_composición_zonas_hetero.xlsx",
            "Resultados_PTL/res_40_comp_hetero.xlsx",
        ),
        (
            "Data_PTL/Data_60_Salidas_composición_zonas_hetero.xlsx",
            "Resultados_PTL/res_60_comp_hetero.xlsx",
        ),
        (
            "Data_PTL/Data_60_Salidas_composición_zonas_homo.xlsx",
            "Resultados_PTL/res_60_comp_homo.xlsx",
        ),
        (
            "Data_PTL/Data_80_Salidas_composición_zonas_hetero.xlsx",
            "Resultados_PTL/res_80_comp_hetero.xlsx",
        ),
        (
            "Data_PTL/Data_80_Salidas_composición_zonas_homo.xlsx",
            "Resultados_PTL/res_80_comp_homo.xlsx",
        ),
    ]
    for file_path, output_file in tasks:
        result, position_zone = solve_ptl_heuristic(file_path)

        # Mostrar resultados en consola
        mostrar_resultados(result, position_zone)

        # Generar archivo Excel con las asignaciones
        generar_excel_asignaciones(result, output_file)
