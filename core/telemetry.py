import math

RULES = {"max_state_age_s": .5, "emergency_distance_m": 20., "risk_horizon_s": 5.,
         "giveway_tcpa_s": 15., "giveway_dcpa_m": 40., "tracking_band_m": [20., 40.],
         "yaw_threshold_dps": 1., "acceleration_threshold_mps2": .2,
         "max_adjacent_delivery_gap_s": .3, "source_clock": "host_delivery_not_verified_simulation_time"}


def wrap(x): return (x+180.) % 360.-180.


def valid_state(state):
    return isinstance(state, dict) and state.get("age_s", 99) <= RULES["max_state_age_s"] and all(
        isinstance(state.get(k), (int, float)) and math.isfinite(state[k])
        for k in ("longitude", "latitude", "heading_deg", "speed_mps"))


def displacement(ego, lon, lat):
    return ((lon-ego["longitude"])*111320*math.cos(math.radians(ego["latitude"])),
            (lat-ego["latitude"])*111320)


def velocity(speed, heading):
    return speed*math.sin(math.radians(heading)), speed*math.cos(math.radians(heading))


def contact(ego, target, ident):
    x, y = displacement(ego, target["longitude"], target["latitude"])
    ev = velocity(ego["speed_mps"], ego["heading_deg"])
    tv = velocity(target.get("speed_mps", 0), target.get("heading_deg", 0))
    vx, vy = tv[0]-ev[0], tv[1]-ev[1]
    vv = vx*vx+vy*vy
    tcpa = -(x*vx+y*vy)/vv if vv > .01 else 0.
    future = max(0., min(5., tcpa))
    distance = math.hypot(x, y)
    return {"id": str(ident), "distance": distance, "bearing": wrap(math.degrees(math.atan2(x,y))-ego["heading_deg"]),
            "tcpa": tcpa, "dcpa": math.hypot(x+vx*max(0.,tcpa), y+vy*max(0.,tcpa)),
            "min_range_5s": math.hypot(x+vx*future, y+vy*future), "speed": target.get("speed_mps",0)}


def contacts(row, ego):
    found = {}
    for ident, state in row.get("states", {}).items():
        if str(ident) != "1" and valid_state(state): found[str(ident)] = contact(ego,state,ident)
    for target in row.get("radar", []):
        ident = str(target.get("target_id"))
        if ident not in found and valid_state(target): found[ident] = contact(ego,target,ident)
    return list(found.values())


def route_command(ego, scenario):
    routes = scenario.get("routes", [])
    route = routes[0] if routes else []
    if not route: return ego["heading_deg"], 5., False
    # Look ahead along the segment nearest the current position; no future row.
    points = [displacement(ego,p[2],p[3]) for p in route]
    if len(points) == 1:
        x,y=points[0]; return math.degrees(math.atan2(x,y)), route[0][4], math.hypot(x,y)<20
    best = None
    for j,(a,b) in enumerate(zip(points[:-1],points[1:])):
        dx,dy=b[0]-a[0],b[1]-a[1]; length=max(1e-9,math.hypot(dx,dy))
        t=max(0.,min(1.,-(a[0]*dx+a[1]*dy)/(length*length)))
        distance=math.hypot(a[0]+t*dx,a[1]+t*dy)
        look=min(1.,t+35./length)
        candidate=(distance,a[0]+look*dx,a[1]+look*dy,j)
        if best is None or distance<best[0]: best=candidate
    _,x,y,j=best
    return math.degrees(math.atan2(x,y)), float(route[min(j+1,len(route)-1)][4]), math.hypot(*points[-1])<20


def infer(row, scenario):
    ego=row.get("states",{}).get("1")
    if not valid_state(ego): return None
    nearby=contacts(row,ego)
    close=min(nearby,key=lambda c:c["distance"]) if nearby else None
    risk=float(any(c["min_range_5s"]<20 for c in nearby)) if nearby else None
    speed=ego["speed_mps"]; yaw=ego.get("yaw_rate_dps"); accel=ego.get("acceleration_mps2")
    if not isinstance(yaw,(float,int)) or not isinstance(accel,(float,int)) or not math.isfinite(yaw+accel):
        executed=None
    elif speed<.2: executed=1
    elif yaw>1.: executed=3
    elif yaw< -1.: executed=2
    elif accel>.2: executed=4
    elif accel< -.2: executed=5
    else: executed=0
    confidence=max(.4,1.-ego.get("age_s",0))
    if close and (close["distance"]<20 or (0<close["tcpa"]<2 and close["dcpa"]<20)):
        action,reason=1,"emergency_separation_rule"
    elif scenario["task"]=="tracking":
        leader=next((c for c in nearby if c["id"]=="2"),None)
        if leader is None: return None
        desired=max(0.,leader["speed"]+max(-1.5,min(1.5,(leader["distance"]-30)*.08)))
        if leader["distance"]<25 or speed>desired+.4: action,reason=5,"tracking_brake"
        elif abs(leader["bearing"])>8: action,reason=(3 if leader["bearing"]>0 else 2),"tracking_bearing"
        elif speed<desired-.4: action,reason=4,"tracking_speed_up"
        else: action,reason=0,"tracking_hold"
    else:
        threats=[c for c in nearby if 0<c["tcpa"]<15 and c["dcpa"]<40 and abs(c["bearing"])<112.5]
        if threats:
            threat=min(threats,key=lambda c:c["tcpa"])
            if threat["bearing"]>=-6: action,reason=3,"starboard_giveway_rule"
            elif threat["tcpa"]<5: action,reason=5,"port_contact_brake"
            else: action,reason=0,"stand_on_heuristic"
        else:
            heading,desired,at_end=route_command(ego,scenario); error=wrap(heading-ego["heading_deg"])
            if at_end: action,reason=1,"route_end_stop"
            elif abs(error)>8: action,reason=(3 if error>0 else 2),"route_heading"
            elif speed>desired+.4: action,reason=5,"route_speed_down"
            elif speed<desired-.4: action,reason=4,"route_speed_up"
            else: action,reason=0,"route_hold"
    return {"target":action,"executed_proxy":executed,"risk_proxy":risk,"confidence":confidence,
            "rule":reason,"nearest":close}
