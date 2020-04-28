# Scripts for parsing MLPerf power measurement logs. 


# Dependencies

Developed under Python 3.

The graphing feature uses plotly.
To install:
```
  pip install plotly==4.6.0
```

Timezone adjustment features uses pytz
To install:
```
  pip install pytz
```

# Script In-Line Paramters

Inside the parser script are some global variables/options.

The following variables are for timing offsets between Host (PTDaemon, usually uses local time) and DUT (usually in UTC).
```
  g_power_tz               = None # pytz.timezone( 'US/Pacific' )
                             # Refer to pytz for timezone list.  This sets the timezone for the Host system
                             
  g_loadgen_tz             = None # pytz.utc
                             # Refer to pytz for timezone list.  This sets the timezone for the DUT system.  Typically does not need to be set.
                             
  g_power_add_td           = timedelta(seconds=3600)
                             # This parameter will add the variable of seconds to the timedelta between loadgen and powerlog timestamps
                             
  g_power_sub_td           = timedelta(seconds=0)
                             # This parameter will subtract the variable of seconds to the timedelta between loadgen and powerlog timestamps
                             # This is b/c timedelta does not use negatives in the seconds place, only in the days place.
```

The following variables are for modifying the graphing and statistical windows.
```
  g_power_window_before_td = timedelta(seconds=30)
  g_power_window_after_td  = timedelta(seconds=30)
                             # These parameters will collate additional data the variable of seconds BEFORE loadgen's BEGIN time
                             # and the variable of seconds AFTER loadgen's END time
                             
  g_power_stats_begin_td   = timedelta(seconds=3)    # not implemented yet
  g_power_stats_end_td     = timedelta(seconds=3)    # not implemented yet
```

# Command-line Parameters

```
  -lgi, --loadgen_in  : Directory of loadgen log files to parse
  -pli, --power_in    : PTDaemon power log file to parse
  -lgo, --loadgen_out : loadgen CSV output filename to write to
  -plo, --power_out   : power log CSV output filename to write to
  -g,   --graph       : graph the data, if possible.  Uses lgo and plo filenames as input
```

# Future plans

- Incorporate Dash (from plotly) for better presentation of data and statistics
- Maybe move global variables to command-line
