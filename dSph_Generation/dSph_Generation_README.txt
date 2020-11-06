There are two versions of the simulation software Genesis.py and Genesis Burkert.py.
Genesis.py produces a database of simulated dSph galaxies with a random selection
of Burkert DM density proﬁles of NFW DM density proﬁles. The manipulation of
the underlying parameters can be done in the if name ==’ main ’ section, begin-
ning at line 318, of the software, where the variable num galaxy (line 321) determines
the number of galaxies that will be produced and can be set to the desired num-
ber. The Burkert proﬁle parameters are governed by the arrays r core array (line
326) and rho central array (line 327) which determine the core radius and central
halo density distributions respectively. The NFW proﬁle parameter distributions are
deﬁned by c param array (line 330) and M 200 array (line 331) which determine the
concentration parameter and the virial mass distributions respectively. These free
parameters can be adjusted to produce the galaxy proﬁles desired. The anisotropy
of the galaxies is controlled by the variable beta array (line 334), and the distance
to the galaxy is deﬁned by the distribution contained in the variable D array (line
335), all of which can be adjusted to suit the needs of the simulation. Moving into
the for loop on line 337, the various parameters are for the stellar density, luminosity,
anisotropy, and distance to the star are deﬁned in the ﬁrst lines of the loop and can
be adjusted appropriately. The variable N on line 344 deﬁnes the number of stars
that are produced in each galaxy and can be set to the desired number. Lines 353-358
randomly choose either a Burkert DM density proﬁle or an NFW DM density proﬁle,
and the following if statements after the proﬁle is selected are where the functions
to generate the galaxies are called. The variable save path on line 397 establishes
the path to the folder where the .csv ﬁles containing the galaxies will be stored and
can be adjusted accordingly. The variable save location on line 403 creates the .csv
ﬁles where the data will be saved and the ﬁlepath should be adjusted similarly to
the save path ﬁlepath. The lines following write the data to the .csv ﬁle and save
the data. To run the simulation software simply adjust the above parameters to the
desired values or ranges and run the software in an IDE or from the command line.