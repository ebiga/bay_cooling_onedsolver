This is simplified 1d fluid dynamics solver to estimate inlet and outlet areas due to ventilation and cooling demands.

Define in file inputs.py:
- Atmospheric and flight conditions
- Inlet position (for boundary layer)
- Outlet: position, type and local pressure coefficient
- Sequence of piping and cooling/ventilation demands

Piping types: pipe, bend [as many as needed, in sequence]
Consumer types: VentingBay, CoolingBay, FanCooler

Modelling details:
- mostly incompressible flow inside the piping, assuming it decelerates from transonic flight to low Mach internal
- the NACA inlet is modelled loosely following ESDU 86002
- inlet and outlet efficiencies are corrected for local boundary layer effects
