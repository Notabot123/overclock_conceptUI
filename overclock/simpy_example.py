import random
import pandas as pd
import simpy
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
SortieLen_MEAN = 3         # Avg. sortie time in hours
SortieLen_SIGMA = 0.5      # Sigma of processing time

MTTF = 1000.0   # Mean time to failure in hours
BREAK_MEAN = 1 / MTTF  # Param. for expovariate distribution
REPAIR_TIME = 30.0     # Time it takes to repair a machine in minutes
JOB_DURATION = 30.0    # Duration of other jobs in hours
NUM_Aircraft = 12      # Number of Aircraft  in the fleet
YEARS = 5 # CONTRACT LENGTH
TOTAL_WEEKS = 52*YEARS              # Simulation time in weeks
SIM_TIME = TOTAL_WEEKS * 7 * 24  # Simulation time in hours

# BER_Prob = 0.2
WEEKS = 14
LEAD_TIME =  WEEKS * 7 * 24 
LRI_REPAIR_TIME =  5 * 24    # Hours  for the LRI to be repaired on-site, after aircraft removal

UOR_TIME = 1 # superman to the rescue £££

item_cost = 10000 # keep constant for simpliciy across SRIs, same with lead and repair times

qualthresh = 75 # at what point its beyond repair in degradation curve

pmthresh = qualthresh + 5 # when should predictive algorithm take course

""" Make a system with subcomponent reliability block diagram """
name = ["system_BD_001","subsystem_BD_001-01","subsystem_BD_001-02","subsystem_BD_001-03","subsystem_BD_001-04"]
levels = [0,1,1,1,1]
fails = np.array([10,50,100,100,740])
df = pd.DataFrame(list(zip(name,levels,fails)),columns=['System','Level','perMillion'])
df['Rate'] = df.perMillion / 1000000
df['Mtbf'] = 1/df.Rate
df['cum_prc'] = np.cumsum(df.perMillion/np.sum(df.perMillion))
df['quality'] = 100
df['lead_time']=LEAD_TIME
df['repair_time']=LRI_REPAIR_TIME
df['item_cost']=item_cost

meansparesdemand = (df['Rate']*df['lead_time'])
df['stock'] = poisson.ppf(0.99, meansparesdemand) # to be optimised
df['mean_spares']= meansparesdemand 
df['reorder_size']= poisson.ppf(0.99, meansparesdemand) - poisson.ppf(0.8, meansparesdemand) # to be optimised
df['reorder_point'] = poisson.ppf(0.8, meansparesdemand) + 1 # to be optimised
df.head()


# supply chain can only process one order at once hence lead time sum not mean.
# it doesn't account for local repairs being much quicker i.e  worst case all new buys
x = np.arange(poisson.ppf(0.01, df['Rate'].sum()*df['lead_time'].sum()),
              poisson.ppf(0.99, df['Rate'].sum()*df['lead_time'].sum()))
plt.plot(x, poisson.pmf(x, df['Rate'].sum()*df['lead_time'].sum()), 'bo', ms=8, label='poisson pmf')
poisson.ppf(0.8, df['Rate'][0]*df['lead_time'][0])
plt.title('Probability of Failures within Order windows')


def time_per_sortie():
    """Return actual processing time for a mission."""
    return np.maximum(random.normalvariate(SortieLen_MEAN, SortieLen_SIGMA),0.4)

def time_to_failure():
    """Return time until next failure for an aircraft."""
    return random.expovariate(df['Rate'].sum())

def time_to_reorder_anewone(i):
    """Return time for industry new buy."""
    return random.expovariate(1/df['lead_time'][i])

def time_to_local_repair_LRI(i,maxall=False):
    """Return time for workshop repair."""
    if maxall: # parallel repair conducted via prognostic recommendation of quality
        r = random.expovariate(1/max(df['repair_time'][i]))
    else:
        r = random.expovariate(1/df['repair_time'][i])
    return r

def which_failed(df):
    r = np.random.rand()
    idx = np.cumsum(1 * r <= df.cum_prc)
    idx[idx>1]=0
    idx = np.argmax(idx)
    return idx

class LRI_Stock(object):
    """
    """
    def __init__(self,env,name,inventory=df):
        self.env = env
        self.name = name
        self.newbuys = 0
        self.uors = 0
        self.preempts = 0
        self.reordevent = self.env.event()
        self.repevent = self.env.event()
        self.msgs = []
        self.costs = 0
        self.log_items = []
        self.log_start = []
        self.log_end = []
        self.inv = np.array(inventory['stock'])
        self.reorder_size = inventory['reorder_size']
        self.reorder_point = inventory['reorder_point']
        self.item_cost = inventory['item_cost']
        self.stocklevels = np.vstack((self.inv,self.inv)) 
        # needs >1 row to be indexed -1 and stacked hereafter, hence vstack

    
    def reorder(self,idx):
        if self.inv[idx]<=self.reorder_point[idx]:
            leadtime = time_to_reorder_anewone(idx)
            start = self.env.now
            yield self.env.timeout(leadtime)
            self.inv[idx] += self.reorder_size[idx]
            self.updatestock_tracker(idx,self.reorder_size[idx]) # for plots later
            self.costs += self.item_cost[idx] * self.reorder_size[idx]
            self.newbuys += self.reorder_size[idx]
            self.reordevent.succeed()
            self.logtimes(idx,start) # for gantt chart later
            print("reorder snail pace: " + str(start) + ":"+ str(self.env.now))
            self.reordevent = self.env.event()

        
    
    def local_repair(self,idx):
        if np.size(idx) > 1: # requesting parallel repairs full sysem overhaul
            print('pre emptive repair, items in parallel: ' + str(np.size(idx)))
            leadtime = time_to_local_repair_LRI(idx,True) # maxall = True, return max repair
            self.preempts += 1
        else:
            leadtime = time_to_local_repair_LRI(idx)   
        start = self.env.now
        yield self.env.timeout(leadtime)
        self.inv[idx] += 1
        self.updatestock_tracker(idx,1)
        self.repevent.succeed()
        if np.size(idx) > 1: # if in paralle, log all.
            for i in idx:
                self.logtimes(i,start) # in case parallel, multi
        else:
            self.logtimes(idx,start)
        msg = "local repair: " + str(start) + ":"+ str(self.env.now)
        self.msgs.append(msg)
        print(msg)
        self.repevent = self.env.event()
    
    def decrement(self,idx):        
        self.inv[idx] -= 1
        self.updatestock_tracker(idx,-1)
        
    def updatestock_tracker(self,idx,num):
        stk = self.stocklevels[-1,:]
        stk[idx] += num
        self.stocklevels = np.vstack((self.stocklevels,stk))
        
    def logtimes(self,idx,start):
        self.log_items.append(idx)
        self.log_start.append(start)
        self.log_end.append(self.env.now)

            

class Aircraft(object):
    """

    """
    def __init__(self, env, name, repairman, stock, uor, df, qualthresh, thresh = 70, predmainton=True):
        self.env = env
        self.name = name
        self.sorties_flown = 0
        self.broken = False
        self.stockpool = stock
        self.log_availability = []
        self.log_landings = 0
        self.log_firings = 0
        self.log_qual = []
        self.quality = 100
        self.subsystems = df
        self.pm_switch = predmainton
        self.qt = qualthresh
        self.thresh = thresh # threshold for pred maint intervention

        # Start "working" and "break_machine" processes for this aircraft.
        self.process = env.process(self.working(repairman, uor))
        env.process(self.break_machine())
        # env.process(self.check_stock())

    def working(self, repairman, uor):
        """Fly missions as long as the sim runs.

        While doing so, the aircraft may break and will request a repairman when this happens.

        """
        while True:
            # Start flying missions / being available
            done_in = time_per_sortie()
            while done_in:
                try:
                    # Working on the sortie
                    start = self.env.now
                    yield self.env.timeout(done_in)
                    self.log_availability.append(1)
                    self.log_landings += (random.randint(1,3))
                    self.log_firings += (random.randint(1,500))
                    self.quality -= random.expovariate(0.1/(done_in*df['Rate'].sum()))
                    self.log_qual.append(self.quality)
                    
                    
                    done_in = 0  # Set to 0 to exit while loop.
                    
                    if self.pm_switch == True:
                        """ if pred maint is switched on, perform preventive maint on all items
                        based upon overall system quality """
                        if self.quality <= self.thresh:
                            """ if we wanted sequential repairs """
                            """
                            for i in range(len(self.stockpool.inv)):
                                env.process(self.stockpool.local_repair(i))
                            """
                            """ parallel could work, passing all indices """
                            allidx = np.arange(len(self.stockpool.inv))
                            self.stockpool.decrement(allidx)
                            env.process(self.stockpool.local_repair(allidx))
                            self.quality = 95
                            self.log_qual.append(self.quality)
                            
                    

                except simpy.Interrupt:
                    self.broken = True
                    self.log_availability.append(0)
                    done_in -= self.env.now - start  # How much time left?
                    
                    idx = which_failed(self.subsystems)

                    # Request a repairman. This will preempt its "other_job".
                    with repairman.request(priority=1) as req:
                        self.stockpool.decrement(idx)
                        yield req
                        yield self.env.timeout(REPAIR_TIME)
                    if self.stockpool.inv[idx] > 0: # if you got it, flaunt it.
                        if self.quality <= self.qt: # If its not fixable beyond quality threshold
                            env.process(self.stockpool.reorder(idx))
                            self.stockpool.costs += 50 # order cost. scale of items handled in class method
                            self.quality = 100 # new one please
                            self.log_qual.append(self.quality)
                        else:
                            env.process(self.stockpool.local_repair(idx))
                            # self.stockpool.updatestock_tracker(idx,1)
                            self.stockpool.costs += 1000
                            self.quality = 95
                            self.log_qual.append(self.quality)

                    else: # yikes we're going to have a back-order!
                        num = self.stockpool.reorder_size[idx]
                        self.stockpool.inv[idx] += num                        
                        self.stockpool.costs += 15000 + self.stockpool.item_cost[idx] * num 
                        self.stockpool.uors += 1
                        self.stockpool.logtimes(idx,start)
                        self.stockpool.updatestock_tracker(idx,num)
                        self.quality = 100 # new one please
                        self.log_qual.append(self.quality)
                        # print("Urgent Operation Request..expedite order at great cost.")

                        with uor.request() as urgent_op:
                            yield urgent_op
                            yield self.env.timeout(UOR_TIME)                           


                    self.broken = False

            #  is done.
            self.sorties_flown += 1

    def break_machine(self):
        """Break the machine every now and then."""
        while True:
            yield self.env.timeout(time_to_failure())
            if not self.broken:
                # Only break the machine if it is currently working.
                self.process.interrupt()
            


def other_jobs(env, repairman):
    """The repairman's other (unimportant) job."""
    while True:
        # Start a new job
        done_in = JOB_DURATION
        while done_in:
            # Retry the job until it is done.
            # It's priority is lower than that of machine repairs.
            with repairman.request(priority=2) as req:
                yield req
                try:
                    start = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start



""" monkey patch """

# https://simpy.readthedocs.io/en/latest/topical_guides/monitoring.html
from functools import partial, wraps

def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation.  The only argument to these functions is the resource
    instance.
    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource)
            # Perform actual operation
            ret = func(*args, **kwargs)
            # Call "post" callback
            if post:
                post(resource)
            return ret
        return wrapper
    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))
def monitor(data, resource):
    """This is our monitoring callback."""
    item = (
        resource._env.now,  # The current simulation time
        resource.count,  # The number of users
        len(resource.queue),  # The number of queued processes
    )
    data.append(item)

""" some plot things """

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


def plotstuff(data, SIM_TIME=SIM_TIME):
    """ plot Maintainer and UOR Resource 
    """

    data = np.stack(data)
    dates = data[:,0]
    newdates = []
    start = dt.datetime.now()
    
    for i,d in enumerate(dates):
        # newdates.append( dt.datetime.fromtimestamp(d) )
        newdates.append( start + dt.timedelta(hours=d))

    newdates = np.stack(newdates)

    plt.plot(data[:,0])
    plt.xticks(rotation=40)
    plt.title('Sim Time')
    plt.show()
    
    plt.plot(newdates,data[:,1])
    plt.title('Utilisation')
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w
    try:
        plt.plot(newdates,moving_average(data[:,1], 20))
    except:
        pass # may be fewer than 20 pts
    plt.xticks(rotation=40)
    plt.show()
    plt.plot(data[:,2])
    plt.title('Queuing')
    plt.show()
    print("total downtime due to queuing: " +  str(np.cumsum(data[:,2])[-1]))


    # Setup and start the simulation
print('Fleet Sim')
random.seed(RANDOM_SEED)  # This helps reproducing the results

# Create an environment and start the setup process
env = simpy.Environment()
repairman = simpy.PreemptiveResource(env, capacity=4)
uor = simpy.Resource(env, capacity=1)

startstock = LRI_Stock(env,'stockpoolAll',df)
fleet = [Aircraft(env, 'Aircraft %d' % i, repairman, startstock, uor, df, qualthresh, pmthresh, True)
            for i in range(NUM_Aircraft)]
data1 = []
data2 = []
monitor1 = partial(monitor, data1)
monitor2 = partial(monitor, data2)
patch_resource(repairman, post=monitor1)  # Patches (only) this resource instance
patch_resource(uor, post=monitor2)  # Patches (only) this resource instance


env.process(other_jobs(env, repairman))

# Execute!
env.run(until=SIM_TIME)

# Analyis/results
print('Aircraft results after %s years' % YEARS)
for aircraft in fleet:
    print('%s flew %d sorties.' % (aircraft.name, aircraft.sorties_flown))
    """ stu added - see we can get whichever bits we want as so """
    print('%s is %d percent available.' % (aircraft.name, (1-aircraft.broken)*100))


plt.style.use('dark_background')
for f in fleet:
    plt.plot(f.log_qual)
plt.show()
np.mean(fleet[0].log_qual)

plotstuff(data1)

try:
    plotstuff(data2)
except:
    print('No urgent requests!')


plt.plot(startstock.log_items)
plt.show()
plt.plot(startstock.stocklevels)
plt.show()


import plotly.express as px
import pandas as pd

def fixdates(dates):
    newdates = []
    start = dt.datetime.now()
    for i,d in enumerate(dates):
        newdates.append( start + dt.timedelta(hours=d))
    newdates = np.stack(newdates)
    return newdates
d1 = fixdates(startstock.log_start) 
d2 = fixdates(startstock.log_end) 
ddf = pd.DataFrame(zip(startstock.log_items,d1,d2),columns=['Task', 'Start', 'Finish'])  

def convert_to_hours(delta):
    total_seconds = delta.total_seconds()
    hours = int(total_seconds // 3600)
    
    return hours
        
        
ddf['delta'] = ((ddf['Finish']-ddf['Start']))
ddf['delta'] = np.digitize(ddf['delta'].apply(convert_to_hours),np.logspace(1.0, 4.0, num=8))

fig = px.timeline(ddf, x_start="Start", x_end="Finish", y="Task", color="delta")
fig.layout.template = 'plotly_dark'
fig.update_yaxes(autorange="reversed")
fig.update_traces(marker_line_width=0.1,marker_line_color="white")
fig.show()