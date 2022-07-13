import os, sys
import pandas as pd
from itertools import groupby, combinations, permutations
from operator import itemgetter
from skyfield.api import load, wgs84, utc, EarthSatellite
from datetime import datetime, timedelta
import numpy as np
from math import *
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None
from sgp4.api import Satrec, WGS72
from scipy.optimize import least_squares

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter3D

mu = 398600     # km**3/s**2
Re = 6378       # km
f = 1/298.26
deg = pi/180
east_long, lat, H = 55.923056, -3.187778, 0.146   # lat deg, long deg, altitude km
phi = east_long*deg                     # latitude rad

# http://servidor.demec.ufpr.br/CFD/bibliografia/Orbital%20Mech.Eng.%20Student-BOOK.pdf

def dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],-a[0]*b[2]+a[2]*b[0],a[0]*b[1]-a[1]*b[0]]

def mag(x):
    return sqrt(sum(i**2 for i in x))

def stumpS(z):
    if z > 0:
        s = (sqrt(z) - sin(sqrt(z)))/(sqrt(z))**3
    elif z < 0:
        s = (sinh(sqrt(-z)) - sqrt(-z))/(sqrt(-z))**3
    else:
        s = 1/6
    return s

def stumpC(z):
    if z > 0:
        c = (1 - cos(sqrt(z)))/z
    elif z < 0:
        c = (cosh(sqrt(-z)) - 1)/(-z)
    else:
        c = 1/2
    return c

def posroot(Roots):
    posroots = abs(Roots[Roots.real>0][Roots[Roots.real>0].imag==0])
    npositive = len(posroots)
    if npositive == 0:
        print('\n\n ** There are no positive roots. \n\n')
        return 0
    # %...If there is more than one positive root, output the
    # %...roots to the command window and prompt the user to
    # %...select which one to use:
    if npositive == 1:
        x = posroots[0]
    else:
        sys.stdout = sys.__stdout__
        # print('\n\n ** There are two or more positive roots.\n')
        # for i in range(npositive):
        #     print("{})  {}".format(i, posroots[i]))
        nchoice = 0#int(input("Choice:    "))
        x = posroots[nchoice]
        sys.stdout = open(os.devnull, 'w')
    return x

def kepler_U(dt, ro, vro, a):
    error = 1.e-8
    nMax = 1000
    x = sqrt(mu)*abs(a)*dt
    n = 0
    ratio = 1
    while abs(ratio) > error and n <= nMax:
        n = n + 1
        C = stumpC(a*x**2)
        S = stumpS(a*x**2)
        F = ro*vro/sqrt(mu)*x**2*C + (1 - a*ro)*x**3*S + ro*x-sqrt(mu)*dt
        dFdx = ro*vro/sqrt(mu)*x*(1 - a*x**2*S)+(1 - a*ro)*x**2*C+ro
        ratio = F/dFdx
        x = x - ratio
    if n > nMax:
        print('**No. iterations of Kepler''s equation')
        print(' =', n)
        print('F/dFdx =\n', F/dFdx)
    return x

def f_and_g(x, t, ro, a):
    z = a*x**2
    f = 1 - x**2/ro*stumpC(z)
    g = t - 1/sqrt(mu)*x**3*stumpS(z)
    return f, g

def gauss(Rho1, Rho2, Rho3, R1, R2, R3, t1, t2, t3):
    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1
    p1 = cross(Rho2,Rho3)
    p2 = cross(Rho1,Rho3)
    p3 = cross(Rho1,Rho2)
    Do = dot(Rho1,p1)
    D = np.array([[dot(R1,p1), dot(R1,p2), dot(R1,p3)], [dot(R2,p1), dot(R2,p2), dot(R2,p3)], [dot(R3,p1), dot(R3,p2), dot(R3,p3)]])
    E = dot(R2,Rho2)
    A = 1/Do*(-D[0,1]*tau3/tau + D[1,1] + D[2,1]*tau1/tau)
    B = (1/6)/Do*(D[0,1]*(tau3**2 - tau**2)*tau3/tau + D[2,1]*(tau**2 - tau1**2)*tau1/tau)
    a = -(A**2 + 2*A*E + np.linalg.norm(R2)**2)
    b = -2*mu*B*(A + E)
    c = -(mu*B)**2
    Roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    x = posroot(Roots)
    f1 = 1 - 1/2*mu*tau1**2/x**3
    f3 = 1 - 1/2*mu*tau3**2/x**3
    g1 = tau1 - 1/6*mu*(tau1/x)**3
    g3 = tau3 - 1/6*mu*(tau3/x)**3
    rho2 = A + mu*B/x**3
    rho1 = 1/Do*((6*(D[2,0]*tau1/tau3 + D[1,0]*tau/tau3)*x**3 + mu*D[2,0]*(tau**2 - tau1**2)*tau1/tau3) /(6*x**3 + mu*(tau**2 - tau3**2)) - D[0,0])
    rho3 = 1/Do*((6*(D[0,2]*tau3/tau1 - D[1,2]*tau/tau1)*x**3 + mu*D[0,2]*(tau**2 - tau3**2)*tau3/tau1) /(6*x**3 + mu*(tau**2 - tau3**2)) - D[2,2])
    r1 = R1 + rho1*Rho1
    r2 = R2 + rho2*Rho2
    r3 = R3 + rho3*Rho3
    v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)
    r_old = r2
    v_old = v2
    rho1_old = rho1
    rho2_old = rho2
    rho3_old = rho3
    diff1 = 1
    diff2 = 1
    diff3 = 1
    n = 0
    nmax = 10
    tol = 1.E-8
    while ((diff1 > tol) & (diff2 > tol) & (diff3 > tol)) & (n < nmax):
        n = n+1
        ro = np.linalg.norm(r2)
        vo = np.linalg.norm(v2)
        vro = dot(v2,r2)/ro
        a = 2/ro - vo**2/mu
        x1 = kepler_U(tau1, ro, vro, a)
        x3 = kepler_U(tau3, ro, vro, a)
        [ff1, gg1] = f_and_g(x1, tau1, ro, a)
        [ff3, gg3] = f_and_g(x3, tau3, ro, a)
        f1 = (f1 + ff1)/2
        f3 = (f3 + ff3)/2
        g1 = (g1 + gg1)/2
        g3 = (g3 + gg3)/2
        c1 = g3/(f1*g3 - f3*g1)
        c3 = -g1/(f1*g3 - f3*g1)
        rho1 = 1/Do*( -D[0,0] + 1/c1*D[1,0] - c3/c1*D[2,0])
        rho2 = 1/Do*( -c1*D[0,1] + D[1,1] - c3*D[2,1])
        rho3 = 1/Do*(-c1/c3*D[0,2] + 1/c3*D[1,2] - D[2,2])
        r1 = R1 + rho1*Rho1
        r2 = R2 + rho2*Rho2
        r3 = R3 + rho3*Rho3
        v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)
        diff1 = abs(rho1 - rho1_old)
        diff2 = abs(rho2 - rho2_old)
        diff3 = abs(rho3 - rho3_old)
        rho1_old = rho1
        rho2_old = rho2
        rho3_old = rho3
    print('\n( **Number of Gauss improvement iterations')
    print(' =)\n\n', n)
    if n >= nmax:
        print('\n\n **Number of iterations exceeds %g \n\n ', nmax);
    r = r2
    v = v2
    return [r, v, r_old, v_old]

def coe_from_sv(R, V):
    eps = 1.e-10
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = dot(R,V)/r
    H = cross(R,V)
    h = np.linalg.norm(H)
    incl = acos(H[2]/h)
    N = cross(np.array([0,0,1]),H)
    n = np.linalg.norm(N)
    if n != 0:
        RA = acos(N[0]/n)
        if N[1] < 0:
            RA = 2*pi - RA
    else:
        RA = 0
    E = 1/mu*((v**2 - mu/r)*R - r*vr*V)
    e = np.linalg.norm(E)
    if n != 0:
        if e > eps:
            w = acos(dot(N,E)/n/e)
            if E[2] < 0:
                w = 2*pi - w
        else:
            w = 0
    else:
        w = 0
    if e > eps:
        TA = acos(dot(E,R)/e/r);
        if vr < 0:
            TA = 2*pi - TA;
    else:
        cp = cross(N,R);
        if cp[2] >= 0:
            TA = acos(dot(N,R)/n/r);
        else:
            TA = 2*pi - acos(dot(N,R)/n/r);
    a = h**2/mu/(1 - e**2)
    coe = [h, e, RA, incl, w, TA, a]
    return coe

def get_angle_amb(angle1,angle2):
    if angle1 < 0:
            angle1m = [pi - angle1, 2*pi + angle1]
    else:
            angle1m = [angle1, pi - angle1]
    if angle2 < 0:
        angle2m = [-angle2, 2*pi + angle2]
    else:
        angle2m = [angle2, 2*pi - angle2]
    for angle in angle2m:
        if abs(angle-angle1m[0]) < 1*10**-12:
            return angle
        elif abs(angle-angle1m[1])<1*10**-12:
            return angle
    return angle1m,angle2m

def determine_orbital_elements(r,rdot):
    print(" ")
    print(" ")
    print("ORBITAL ELEMENTS")
    print(" ")

    # Semi-Major Axis, a
    a = 1/(2/mag(r) - (mag(rdot)**2)/mu)
    print("The semi-major axis equals", a)
    if a < 0: return False

    # Eccentricity of Orbit, e
    e = sqrt(1 - (mag(cross(r,rdot))**2)/(mu*a))
    print("The eccentricity of the asteroid orbit equals", e)

    # Inclination of Asteroid Orbit (i)
    l = cross(r,rdot)
    irads = atan(sqrt(l[0]**2 + l[1]**2)/l[2])
    i = irads*180/pi
    print("The inclination of the asteroid orbit equals", i)

    # Longitude of Ascending Node (o)
    o1 = asin(l[0]/(mag(l)*sin(irads)))
    o2 = acos(-l[1]/(mag(l)*sin(irads)))
    orads = get_angle_amb(o1,o2)
    o = orads*180/pi
    print("The longitude of the ascending node of the asteroid orbit equals", o)

    # Argument of the Perihilion (w)
    U1 = asin(r[2]/(mag(r)*sin(irads)))
    U2 = acos((r[0]*cos(orads)+r[1]*sin(orads))/mag(r))
    U = get_angle_amb(U1,U2)

    v1 = asin((((a*(1-e**2))/mag(l))*(dot(r,rdot)/mag(r)))/e)
    v2 = acos(((a*(1-e**2))/mag(r) - 1)/e)
    v = get_angle_amb(v1,v2)

    wrads = U - v
    w = wrads*180/pi % 360
    print("The argument of the perihilion equals", w)

    # Mean Anomaly (M)
    E = acos((1-mag(r)/a)/e)
    Mrads = E - e*sin(E)
    M = Mrads*180/pi
    print("The mean anomaly equals", M)
    return [a, e, i, o, w, M]

def Run(H, phi, t, ra, dec, theta):
    fac1 = Re/sqrt(1-(2*f - f*f)*sin(phi)**2)
    fac2 = (Re*(1-f)**2/sqrt(1-(2*f - f*f)*sin(phi)**2) + H)*sin(phi)
    R, rho = [], []
    for i in range(3):
        R.append([(fac1 + H)*cos(phi)*cos(theta[i]), (fac1 + H)*cos(phi)*sin(theta[i]), fac2])
        rho.append([cos(dec[i])*cos(ra[i]), cos(dec[i])*sin(ra[i]), sin(dec[i])])
        # R(i,1) = (fac1 + H)*cos(phi)*cos(theta(i))
        # R(i,2) = (fac1 + H)*cos(phi)*sin(theta(i))
        # R(i,3) = fac2
        # rho(i,1) = cos(dec(i))*cos(ra(i))
        # rho(i,2) = cos(dec(i))*sin(ra(i))
        # rho(i,3) = sin(dec(i))

    t1, t2, t3 = t[0],t[1],t[2]
    Rho1,Rho2,Rho3 = np.array(rho[0]),np.array(rho[1]),np.array(rho[2])
    R1,R2,R3 = np.array(R[0]),np.array(R[1]),np.array(R[2])
    [r, v, r_old, v_old] = gauss(Rho1, Rho2, Rho3, R1, R2, R3, t1, t2, t3)
    coe_old = coe_from_sv(r_old,v_old)
    coe = coe_from_sv(r,v)
    elems = determine_orbital_elements(r, v)
    if elems == False or coe[6] < 6000 or coe[0]**2 /mu/(1 + coe[1]) < 6000: raise ValueError('A very specific bad thing happened.')
    elems = [*elems, *r, *v]
    print('---------------------------------------------------')
    #print('Radius of earth (km) =', Re)
    #print('Flattening factor =', f)
    #print('Gravitational parameter (km**3/s**2) =', mu)
    #print('\n\n Input data:\n');
    #print('Latitude (deg) =', phi/deg);
    #print('Altitude above sea level (km) =', H);
    #print('\n\n Observations:')
    #print('Time (s) Right ascension (deg) Declination (deg)')
    #print(' Local sidereal time (deg)')
    #for i in range(3):
    #    print('%9.4g %17.4f %19.4f %23.4f', t[i], ra[i]/deg, dec[i]/deg, theta[i]/deg)
    #print('\n\n Solution:\n')
    print('Without iterative improvement...\n')
    print('\n');
    print('r (km) = [%g, %g, %g]', r_old[0], r_old[1], r_old[2])
    print('v (km/s) = [%g, %g, %g]', v_old[0], v_old[1], v_old[2])
    print('\n')
    print('Angular momentum (km**2/s) =', coe_old[0])
    print('Eccentricity =', coe_old[1])
    print('RA of ascending node (deg) =', coe_old[2]/deg)
    print('Inclination (deg) =', coe_old[3]/deg)
    print('Argument of perigee (deg) =', coe_old[4]/deg)
    print('True anomaly (deg) =', coe_old[5]/deg)
    print('Semimajor axis (km) =', coe_old[6])
    print('Periapse radius (km) =', coe_old[0]**2/mu/(1 + coe_old[1]))
    # If the orbit is an ellipse, output the period:
    if coe_old[1]<1:
        T = 2*pi/sqrt(mu)*coe_old[6]**1.5;
        print('Period:')
        print('Seconds =', T)
        print('Minutes =', T/60)
        print('Hours =', T/3600)
        print('Days =', T/24/3600)
    print('\n\n With iterative improvement...\n')
    print('\n')
    print('r (km) = [%g, %g, %g]', r[0], r[1], r[2])
    print('v (km/s) = [%g, %g, %g]', v[0], v[1], v[2])
    print('\n')
    print('Angular momentum (km**2/s) =', coe[0])
    print('Eccentricity =', coe[1])
    print('RA of ascending node (deg) =', coe[2]/deg)
    print('Inclination (deg) =', coe[3]/deg)
    print('Argument of perigee (deg) =', coe[4]/deg)
    print('True anomaly (deg) =', coe[5]/deg)
    print('Semimajor axis (km) =', coe[6])
    print('Periapse radius (km) =', coe[0]**2 /mu/(1 + coe[1]))
    # If the orbit is an ellipse, output the period:
    if coe[1]<1:
        T = 2*pi/sqrt(mu)*coe[6]**1.5;
        print('Period:')
        print('Seconds =', T)
        print('Minutes =', T/60)
        print('Hours =', T/3600)
        print('Days =', T/24/3600)
    # input('\n-----------------------------------------------\n')

    return elems

def J0(year, month, day):
    return 367*year - np.fix(7*(year + np.fix((month + 9)/12))/4) + np.fix(275*month/9) + day + 1721013.5

def LST(y, m, d, h, min, s, EL):
    ut = h + min/60 + s/3600
    j0 = J0(y, m, d)
    j = (j0 - 2451545)/36525
    g0 = 100.4606184 + 36000.77004*j + 0.000387933*j**2 - 2.583e-8*j**3
    g0 = zeroTo360(g0)
    gst = g0 + 360.98564724*ut/24
    lst = gst + EL
    lst = lst - 360*np.fix(lst/360)
    return lst*deg

def zeroTo360(x):
    if (x >= 360):
        x = x - np.fix(x/360)*360
    elif (x < 0):
        x = x - (np.fix(x/360) - 1)*360
    y = x
    return y

# ----------------------------------------------------------------------------------------------------------------------------------

# Orders RA, Dec, Time array by time increasing by rearranging start end points of each streaklet in collection of streaklets
def GetDirecOrder(mini, i):
    ras=[]
    decs=[]
    times = []
    mini['Time'] = None
    for h in range(len(mini)):
        mini['Time'].iloc[h] = datetime.strptime(mini['Filename'].iloc[h][4:21],'%Y-%m-%d_%H%M%S')
    mini = mini.sort_values(by=['Time'])
    for h in range(len(mini)):
        start = datetime.strptime(mini['Filename'].iloc[h][4:21],'%Y-%m-%d_%H%M%S')
        point = list(map(str.strip, mini['RADecPoint1'].iloc[h].strip('][').replace('"', '').split(',')))
        point = [float(n) for n in point]
        ras.append(point[0])
        decs.append(point[1])
        times.append(start)
        point = list(map(str.strip, mini['RADecPoint2'].iloc[h].strip('][').replace('"', '').split(',')))
        point = [float(n) for n in point]
        end = start + timedelta(seconds=5)
        ras.append(point[0])
        decs.append(point[1])
        times.append(end)
        # plt.plot(ras[-2:],decs[-2:],c='k',label='Observed Streaklets')

    arr = np.vstack((ras,decs,times))
    times.sort()
    # plt.plot(ras,decs,c='b',alpha=0.3)
    if np.diff(ras[::2], n=len(ras[::2])-1) > 0:
        arr = arr[:, arr[0, :].argsort()]
    else:
        arr = arr[:, arr[0, :].argsort()[::-1]]
    arr[2] = arr[2][arr[2].argsort()]

    ras, decs, times = arr[0], arr[1], arr[2]
    # for p in range(len(ras)):
    #     plt.annotate(times[p],(ras[p],decs[p]))
    # plt.plot(ras,decs,c='r',alpha=0.3)
    # plt.annotate  ("{} ({})".format(mini['Satellite'].iloc[0],int(mini['NORAD_CAT_ID'].iloc[0])),(ras[0]+0.5,decs[0]),wrap=True)
    # plt.xlabel("RA (deg)")
    # plt.ylabel("Dec (deg)")
    # plt.title(format(mini['Satellite'].iloc[0]) )
    # plt.savefig("{}.png".format(int(mini['NORAD_CAT_ID'].iloc[0])))
    # plt.show()
    return ras, decs, times

def ElemsToRaDec(elems, satnum, epoch, bstar, ndot, nddot, no_kozai, exp_times, exp_ra_decs):
    satrec = Satrec()
    ecco, argpo, inclo, nodeo, mo = elems[1], elems[4]*pi/180, elems[2]*pi/180, elems[3]*pi/180, elems[5]*pi/180
    satrec.sgp4init(
        WGS72,           # gravity model
        'i',             # 'a' = old AFSPC mode, 'i' = improved mode
        int(satnum), # Satellite number
        epoch, # days since 1949 December 31 00:00 UT
        bstar, # drag coefficient (/earth radii)
        ndot, # ballistic coefficient (revs/day)
        nddot, # second derivative of mean motion (revs/day^3)
        ecco, # eccentricity
        argpo, # argument of perigee (radians)
        inclo, # inclination (radians)
        mo, # mean anomaly (radians)
        no_kozai, # mean motion (radians/minute)
        nodeo # right ascension of ascending node (radians)
    )

    lat, long, elevation = 55.923056, -3.187778, 146    # Royal Observatory, Edinburgh
    ts = load.timescale()
    sat = EarthSatellite.from_satrec(satrec, ts)
    bluffton = wgs84.latlon(lat,long,elevation)
    obs_times = []
    for i in range(len(exp_times)):
        obs_times.append(ts.utc(*exp_times[i]))
    difference = sat - bluffton
    ras, decs, dists = [], [], []
    for i in obs_times:
        topocentric = difference.at(i)
        ra, dec, dist = topocentric.radec()
        ras.append(ra._degrees)
        decs.append(dec._degrees)

    actual_ras_decs = [*ras, *decs]
    # print(actual_ras_decs)
    # print(exp_ra_decs)
    # input(np.array(actual_ras_decs)-np.array(exp_ra_decs))

    return np.array(actual_ras_decs)-np.array(exp_ra_decs)

def ElementsPrediction(elems, satnum, epoch, bstar, ndot, nddot, no_kozai, exp_times):
    satrec = Satrec()
    ecco, argpo, inclo, nodeo, mo = elems[1], elems[4]*pi/180, elems[2]*pi/180, elems[3]*pi/180, elems[5]*pi/180
    satrec.sgp4init(
        WGS72,           # gravity model
        'i',             # 'a' = old AFSPC mode, 'i' = improved mode
        int(satnum), # Satellite number
        epoch, # days since 1949 December 31 00:00 UT
        bstar, # drag coefficient (/earth radii)
        ndot, # ballistic coefficient (revs/day)
        nddot, # second derivative of mean motion (revs/day^3)
        ecco, # eccentricity
        argpo, # argument of perigee (radians)
        inclo, # inclination (radians)
        mo, # mean anomaly (radians)
        no_kozai, # mean motion (radians/minute)
        nodeo # right ascension of ascending node (radians)
    )

    lat, long, elevation = 55.923056, -3.187778, 146    # Royal Observatory, Edinburgh
    ts = load.timescale()
    sat = EarthSatellite.from_satrec(satrec, ts)
    bluffton = wgs84.latlon(lat,long,elevation)
    obs_times = []
    for i in range(len(exp_times)):
        obs_times.append(ts.utc(*exp_times[i]))
    difference = sat - bluffton
    ras, decs, dists = [], [], []
    for i in obs_times:
        topocentric = difference.at(i)
        ra, dec, dist = topocentric.radec()
        ras.append(ra._degrees)
        decs.append(dec._degrees)
    print(ras)
    input(decs)

def dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def GetSemiMajorAxis(exp_ra_decs, exp_times):
    arr = np.array(exp_ra_decs).reshape(2, int(len(exp_ra_decs)/2))
    ras, decs = arr[0], arr[1]
    angle_sep = np.sqrt( ( abs(ras[0]-ras[-1]) )**2 + ( abs(decs[0]-decs[-1]) )**2 )
    t_diff = (datetime(*exp_times[-1]) - datetime(*exp_times[0])).seconds
    ang_vel = angle_sep / t_diff
    print("a",angle_sep)
    period = (360 / ang_vel) / 60
    print(ang_vel)
    print(period)

    units = []
    for i in range(2):
        RA = ras[i]
        Dec = decs[i]
        print([cos(RA)*cos(Dec), sin(RA)*cos(Dec), sin(Dec)])
        units.append([cos(RA)*cos(Dec), sin(RA)*cos(Dec), sin(Dec)])

    angle_sep = dot(units[0],units[1])
    ang_vel = angle_sep / 5
    print("a",angle_sep)
    period = (360 / ang_vel) / 60
    print(period)
    radius = ((period**2 * 6.67E-11 * 5.97E24) / (4 * pi**2))**(1/3)

    input(radius)

def NewFunc(ras, decs, times, mini):
    exp_ra_decs2 = np.array([ras,decs], dtype='object')
    exp_ra_decs = [*ras, *decs]
    exp_times = []
    for i in range(len(times)):
        exp_times.append([times[i].year, times[i].month, times[i].day, times[i].hour, times[i].minute, times[i].second])
    # c = [0, int((len(ras)-1)/2), len(ras)-1]

    combs = [k for k in permutations(np.arange(len(ras))+1, 3)]
    orb_elems = []
    its = 0
    successes = []
    for c in combs:
        if sorted(c) == list(c):
            # print(c)
            ras2, decs2, data = [], [], []
            for cc in c:
                data.append([times[cc-1].year, times[cc-1].month, times[cc-1].day, times[cc-1].hour, times[cc-1].minute, times[cc-1].second])
                ras2.append(ras[cc-1])
                decs2.append(decs[cc-1])
            data.append([*ras2, *decs2])
            data = np.array(data)
            # Make variables for Gauss Method
            t = [datetime(*[int(i) for i in data[x]]) for x in range(3)]             # time in sec
            seconds = [0, (t[1]-t[0]).total_seconds(), (t[2]-t[0]).total_seconds()]
            ra = data[3][0:3]*deg       # ra
            dec = data[3][3:]*deg       # dec
            theta = [LST(*[int(i) for i in data[x]], east_long) for x in range(3)]

            try:
                sys.stdout = open(os.devnull, 'w')
                elems_out = Run(H, phi, seconds, ra, dec, theta)
                orb_elems.append(elems_out)
                successes.append(c)
                sys.stdout = sys.__stdout__
            except:
                sys.stdout = sys.__stdout__
                # print("FAil")
                # return False
            sys.stdout = sys.__stdout__


    tle1 = mini['TLE1'].iloc[0]
    tle2 = mini['TLE2'].iloc[0]
    satnum = mini['NORAD_CAT_ID'].iloc[0]
    date_time = datetime.strftime(t[1],"%Y-%m-%d-%H-%M-%S").split("-")
    init_epoch = [1949, 12, 31, 0, 0, 0]
    diff = datetime(*[int(x) for x in date_time]) - datetime(*init_epoch)
    epoch = diff.days + (diff.seconds/(3600*24))
    tle1 = mini['TLE1'].iloc[0]
    bstar = tle1[53:61]
    bstar = bstar.split("-")
    if bstar[0] == '':
        bstar = float("-."+bstar[1].strip())*10**(-1*float(bstar[2]))
    else:
        bstar = float("."+bstar[0].strip())*10**(-1*float(bstar[1]))
    ndot = tle1[33:43]
    ndot = float(ndot)
    nddot = tle1[44:52]
    nddot = nddot.split("-")
    nddot = float("."+nddot[0].strip())*10**(-1*float(nddot[1]))
    no_kozai = float(tle2[52:63])/229.183

    if successes == []:
        print("  Unable to determine inital orbit")
        return [False], 10000
    xs = []
    costs = []
    for k in range(len(successes)):
        # elems_out[0] = GetSemiMajorAxis(exp_ra_decs, exp_times)
        elems_out = orb_elems[k]
        res_1 = least_squares(ElemsToRaDec, elems_out[0:6], args=(satnum, epoch, bstar, ndot, nddot, no_kozai, exp_times, exp_ra_decs), method='lm', max_nfev=1000)


        satrec = Satrec()
        elems = res_1.x
        ecco, argpo, inclo, nodeo, mo = elems[1], elems[4]*pi/180, elems[2]*pi/180, elems[3]*pi/180, elems[5]*pi/180
        satrec.sgp4init(
            WGS72,           # gravity model
            'i',             # 'a' = old AFSPC mode, 'i' = improved mode
            int(satnum), # Satellite number
            epoch, # days since 1949 December 31 00:00 UT
            bstar, # drag coefficient (/earth radii)
            ndot, # ballistic coefficient (revs/day)
            nddot, # second derivative of mean motion (revs/day^3)
            ecco, # eccentricity
            argpo, # argument of perigee (radians)
            inclo, # inclination (radians)
            mo, # mean anomaly (radians)
            no_kozai, # mean motion (radians/minute)
            nodeo # right ascension of ascending node (radians)
        )

        lat, long, elevation = 55.923056, -3.187778, 146    # Royal Observatory, Edinburgh
        ts = load.timescale()
        sat = EarthSatellite.from_satrec(satrec, ts)
        bluffton = wgs84.latlon(lat,long,elevation)
        obs_times = []
        for i in range(len(exp_times)):
            obs_times.append(ts.utc(*exp_times[i]))
        difference = sat - bluffton
        ras, decs, dists = [], [], []
        for i in obs_times:
            topocentric = difference.at(i)
            ra, dec, dist = topocentric.radec()
            ras.append(ra._degrees)
            decs.append(dec._degrees)

        # plt.scatter(ras,decs,c='b',label='Orbit Fit Predictions')
        # plt.scatter(exp_ra_decs2[0], exp_ra_decs2[1],c='orange')
        # plt.legend()
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())
        # plt.savefig("{}_with_orbit_fit.png".format(int(satnum)))
        # plt.show()
        xs.append(res_1.x)
        costs.append(res_1.cost)

    ind = np.argmin(np.array(costs))
    output_elems = xs[ind]
    return output_elems, costs[ind]

def GaussCombinations(ras, decs, times, mini):
    combs = [k for k in permutations(np.arange(len(ras))+1, 3)]
    orb_elems = []
    its = 0
    for c in combs:
        if sorted(c) == list(c):
            its+=1
            data = []
            ras2, decs2 = [], []
            for cc in c:
                data.append([times[cc-1].year, times[cc-1].month, times[cc-1].day, times[cc-1].hour, times[cc-1].minute, times[cc-1].second])
                ras2.append(ras[cc-1])
                decs2.append(decs[cc-1])
            data.append([*ras2, *decs2])
            data = np.array(data)
            # Make variables for Gauss Method
            t = [datetime(*[int(i) for i in data[x]]) for x in range(3)]             # time in sec
            seconds = [0, (t[1]-t[0]).total_seconds(), (t[2]-t[0]).total_seconds()]
            ra = data[3][0:3]*deg       # ra
            dec = data[3][3:]*deg       # dec
            theta = [LST(*[int(i) for i in data[x]], east_long) for x in range(3)]
            tle1 = mini['TLE1'].iloc[0]
            tle2 = mini['TLE2'].iloc[0]
            satnum = mini['NORAD_CAT_ID'].iloc[0]
            date_time = datetime.strftime(t[1],"%Y-%m-%d-%H-%M-%S").split("-")
            init_epoch = [1949, 12, 31, 0, 0, 0]
            diff = datetime(*[int(x) for x in date_time]) - datetime(*init_epoch)
            epoch = diff.days + (diff.seconds/(3600*24))
            tle1 = mini['TLE1'].iloc[0]
            bstar = tle1[53:61]
            bstar = bstar.split("-")
            bstar = float("."+bstar[0].strip())*10**(-1*float(bstar[1]))
            ndot = tle1[33:43]
            ndot = float(ndot)
            nddot = tle1[44:52]
            nddot = nddot.split("-")
            nddot = float("."+nddot[0].strip())*10**(-1*float(nddot[1]))
            no_kozai = float(tle2[52:63])/229.183
            date_times = [[t[0].year, t[0].month, t[0].day, t[0].hour, t[0].minute, t[0].second], [t[1].year, t[1].month, t[1].day, t[1].hour, t[1].minute, t[1].second], [t[2].year, t[2].month, t[2].day, t[2].hour, t[2].minute, t[2].second]]
            green = 0
            try:
                sys.stdout = open(os.devnull, 'w')
                elems_out = Run(H, phi, seconds, ra, dec, theta)
                sys.stdout = sys.__stdout__
                green = 1
            except Exception as e:
                sys.stdout = sys.__stdout__
            if green == 1:
                sys.stdout = sys.__stdout__
                # print(elems_out[0:6])
                # res_1 = least_squares(ElemsToRaDec, elems_out[0:6], args=(satnum, epoch, bstar, ndot, nddot, no_kozai, date_times))#, method='lm')
                # input(res_1.x)
                orb_elems.append(elems_out)
                sucess += 1
                main = 1
                # print(e)

            if main == 50:
                attempts += 9999
                for k in range(9999):
                    sec_dist = [abs(np.random.normal(loc=seconds[i], scale=0.33, size=1)).tolist()[0] for i in range(3)]
                    ra_dist = [np.random.normal(loc=ra[i], scale=0.1, size=1).tolist()[0] for i in range(3)]
                    dec_dist = [np.random.normal(loc=dec[i], scale=0.1, size=1).tolist()[0] for i in range(3)]
                    # Run Gauss method
                    try:
                        sys.stdout = open(os.devnull, 'w')
                        orb_elems.append(Run(H, phi, sec_dist, ra_dist, dec_dist, theta))
                        sucess += 1
                    except Exception as e:
                        sys.stdout = sys.__stdout__
                        # print(e)
            sys.stdout = sys.__stdout__
            print("{}) {}/{}".format(its,sucess,attempts))
            print(len(orb_elems))

    input("loop end")
    return orb_elems

def GetAllObservations():
    frame = OrbitPlotter3D()
    frame.set_attractor(Earth)
    a = os.listdir("/home/s1901554/Documents/SpaceTrafficManagement")
    b = pd.DataFrame()
    for i in a:
        if "output_data" in i:
            a = pd.read_csv("/home/s1901554/Documents/SpaceTrafficManagement/"+i)
            a['Date'] = i[12:-4]
            b = pd.concat([b,a])
    b = b[b['Date']=="2022-05-11"]
    fails = b[b['Satellite'].astype(str)=="FAILED"].reset_index(drop=True)
    sucesses = b[b['Satellite'].astype(str)!="FAILED"].reset_index(drop=True)

    yes, no = 0, 0
    for i in np.unique(sucesses['NORAD_CAT_ID']):
        mini = sucesses[sucesses['NORAD_CAT_ID']==i]
        if len(mini) >= 3:
            ras, decs, times = GetDirecOrder(mini, i)
            elems, cost = NewFunc(ras, decs, times, mini)
            if len(elems) == 6 and cost <= 500:
                print("Orbit fit success for {} ({}) with cost {}".format(mini['Satellite'].iloc[0],int(mini['NORAD_CAT_ID'].iloc[0]),cost))
                epoch = Time(times[0])
                mn = elems
                if mn[2] > 180:
                    while mn[2] > 180:
                        mn[2] -= 180
                elif mn[2] < 0:
                    while mn[2] < 0:
                        mn[2] += 180
                orb2 = Orbit.from_classical(Earth, mn[0]*u.km, mn[1]*u.one, mn[2]*u.deg, mn[3]*u.deg, mn[4]*u.deg, mn[5]*u.deg, epoch)
                fig = frame.plot(orb2, label="{} ({})".format(mini['Satellite'].iloc[0],mini['NORAD_CAT_ID'].iloc[0]))
                yes += 1
                df = pd.read_csv("/home/s1901554/Documents/SpaceTrafficManagement/output_data_2022-05-10.csv")
                df = df[df['NORAD_CAT_ID'] == 1807]
                input(df)
                print(df['RADecPoint1'],df['RADecPoint2'])
                tle1 = df['TLE1'].iloc[0]
                tle2 = df['TLE2'].iloc[0]
                satnum = df['NORAD_CAT_ID'].iloc[0]
                date_time = datetime.strptime(df['Filename'].iloc[0][4:21],"%Y-%m-%d_%H%M%S")
                init_epoch = [1949, 12, 31, 0, 0, 0]
                diff = date_time - datetime(*init_epoch)
                epoch = diff.days + (diff.seconds/(3600*24))
                bstar = tle1[53:61]
                bstar = bstar.split("-")
                if bstar[0] == '':
                    bstar = float("-."+bstar[1].strip())*10**(-1*float(bstar[2]))
                else:
                    bstar = float("."+bstar[0].strip())*10**(-1*float(bstar[1]))
                ndot = tle1[33:43]
                ndot = float(ndot)
                nddot = tle1[44:52]
                nddot = nddot.split("-")
                nddot = float("."+nddot[0].strip())*10**(-1*float(nddot[1]))
                no_kozai = float(tle2[52:63])/229.183
                datetime2 = date_time + timedelta(seconds=5)
                exp_times=[]
                exp_times.append([date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, date_time.second])
                exp_times.append([datetime2.year, datetime2.month, datetime2.day, datetime2.hour, datetime2.minute, datetime2.second])
                ElementsPrediction(elems, satnum, epoch, bstar, ndot, nddot, no_kozai, exp_times)

            else:
                print("Orbit fit failed for {} ({}) with cost {}".format(mini['Satellite'].iloc[0],int(mini['NORAD_CAT_ID'].iloc[0]),cost))
                no += 1

    # plt.xlabel("RA (deg)")
    # plt.ylabel("Dec (deg)")
    # plt.title("Detected streaklets of three satellites")
    # plt.savefig("MultipleSatellites.png")
    # plt.show()

    print("\nNumber of successful orbit fits: {}/{}".format(int(yes), int(no)))
    fig.show()
    exit()

    #fails = sucesses
    fails['IMG'] = None
    for i in range(len(fails)):
        fails['IMG'][i] = int(fails['Filename'][i][28:32])
    data = fails['IMG'].to_list()
    multiple_obs = []
    single_obs = []
    for k, g in groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
        b = list(map(itemgetter(1), g))
        if len(b) > 1: multiple_obs.append(b)
        else: single_obs.append(b)
    for i in multiple_obs:
        i = [int(k) for k in i]
        if len(i)>2:
            print(i)
            times = []
            mini = fails[fails['IMG'].isin(i)].reset_index(drop=True)
            input(mini[['Satellite','IMG']])
            ras, decs, times = GetDirecOrder(mini, i)
            orb_elems = GaussCombinations(ras, decs, times)
            input("end of sat.")

GetAllObservations()
