This is simplified 1d fluid dynamics solver to estimate inlet and outlet areas due to ventilation and cooling demands.

Define in file inputs.py:
- Atmospheric and flight conditions
- Inlet position (for boundary layer)
- Outlet: position, type and local pressure coefficient
- Sequence of piping and cooling/ventilation demands

Piping types: pipe, bend [as many as needed, in sequence]
Consumer types: VentingBay, CoolingBay, FanCooler

Modelling details:
- compressible formulation for the piping elements
- the NACA inlet is modelled similarly to ESDU 86002, incl. local boundary layer effects
- outlet models are based on NACA TN3466
