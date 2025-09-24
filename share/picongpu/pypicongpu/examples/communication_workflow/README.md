# Communication Workflow Example

This is a small example sketching out
what use cases and benefits the communication with a database could bring.
This is a very rough, exploratory draft at the moment:
In particular, not all interfaces and responsibilities are well defined by now.
Larger-scale workflows would probably not happen from within a single python script
but would employ a workflow engine to orchestrate different simulation steps.
What that interaction would look like in detail is yet to be explored.

## What it does

### Abstractly

Abstractly, this workflow runs simulations varying some parameters
and utilises the information from the metadata database
to do some post-processing on them.

It highlights how we can:
- record (default, later being standardised) metadata in a database
- record additional information during and after the simulation
- extract metadata in a structured manner
- access data from disk via information from the metadata

### Concretely

Concretely, this code sets up a very simple foil (just rectangular wall) and a plane-wave laser
and varies their wall width and laser duration.
As it uses a free-formula density,
the height and the width of the foil are not easily extractable
from the default metadata (holding only the C++ code expression).
This problem is solved by recording them as additional information
when setting up the simulations.

After all simulations have run,
the post-processing contains two steps:
- Compute the total mass of electrons in the simulation
  and store this in the metadata after the fact
- Extract the electron spectrum from the simulation results
  by querying the metadata for its location
  and plot this

## Implementation details

The `main.py` top-level file runs the simulation.
It refers to the `infrastructure` for most of its details.
This separates concerns and seems to be a good idea in general.

But in the current context, it's most notable
that this structure makes the `COMMUNICATOR` easily swappable.
This object will likely contain personal user information
(like responsible authors, projects, etc.),
site-specific details (like how to reach the database)
and might potentially even include authentication secrets.
So, it could be a standard workflow to NOT include this file
when sharing with colleagues or in a data publication.

The `Communicator` is supposed to be
the composable interface for interaction with the database
including the configuration and appropriate injection of
user-/project-specific information.
Its interface, responsibilities and functionalities
are not at all worked out in detail, let alone properly defined.
That's simply because I didn't know
what interfaces and customisation points could be of use.
Its functionality is situated somewhere between
the PyPIConGPU-internal `SimulationLogger`
and interchangeable database interfaces defined in `piccom`.

In that sense, everything regarding the `Communicator`
is quite hacky, uses internal knowledge
and certainly does not constitute an example of
what production code using this would look like.
