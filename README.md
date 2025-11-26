# SIMULACION-DINAMICA

Simulador visual estilo "videojuego" para la droguería MENAR usando SimPy + Pygame.

Requerimientos y ejecución (Windows PowerShell):

1. Crear/activar el entorno virtual (opcional pero recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Ejecutar la simulación:

```powershell
python .\simulacion.py
```

Notas:
- Coloca `icono_usuario.png` en el mismo directorio que `simulacion.py` para que se muestre el icono de los clientes. Si falta la imagen, se usa un dibujo circular como fallback.
- Ajusta `CLIENT_RADIUS` o `ICON_SIZE` (en `simulacion.py`) para cambiar el tamaño visual del icono/cliente.
- Si usas datos reales desde Excel, coloca `datos_drogueria.xlsx` junto al script (el repo ya contiene un ejemplo si se incluyó).

Si quieres que agregue ejemplos de parámetros o capture de pantalla, dime y lo agrego.
# SIMULACION-DINAMICA