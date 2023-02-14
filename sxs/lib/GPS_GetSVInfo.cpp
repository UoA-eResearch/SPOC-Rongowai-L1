/*
* Copyright (c) 2011, Scott Gleason
* All rights reserved.
*
* Written by Scott Gleason
* Modified by X. Lin
* 1) revise the input and output structures to data array
* 2) interfacing inputs/outputs with matlab mex function
* 3) inputs: prn, gps week and seconds since start of gps week, sp3-d filename
* 4) outputs: satellite's ECEF positions and velocity velocity vectors, clk error and drifts
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the authors' names nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "intrpsp3c.h"

#define SUCCESS 100
#define FAILURE 101

using namespace std;
double sat_pos[8];

int GetSVInfo(unsigned int prn, unsigned int week, double TOW, double sat_pos[], char* infileXname)
{
    int retval = SUCCESS;
    char prnNum[10];
    string prnNum_str;
    DateTime currEpoch;
    double PosVel[8];
    string orbfile;
    GPSTime x;
    SP3cFile mysp3;    

    // this hack prevents the occasional crash on array copying (on some platforms)
    sprintf(prnNum, "G%2d", prn);
    if (prn < 10)
        prnNum[1] = '0';
    for (int i = 0; i < 3; i++)
        prnNum_str += prnNum[i];

    x.GPSWeek = week;
    x.secsOfWeek = TOW;
    currEpoch = DateTime(x);

    mysp3.setPathFilenameMode(infileXname);
    mysp3.readHeader();

    retval = (int)mysp3.getSVPosVel(currEpoch, prnNum_str, PosVel);
    
    if (retval == 0)
    {
        sat_pos[0] = PosVel[0] * 1000;  // m
        sat_pos[1] = PosVel[1] * 1000;
        sat_pos[2] = PosVel[2] * 1000;
        sat_pos[3] = PosVel[3] / 1e6;   // sec

        sat_pos[4] = PosVel[4] * 1000;  // m/s
        sat_pos[5] = PosVel[5] * 1000;
        sat_pos[6] = PosVel[6] * 1000;
        sat_pos[7] = PosVel[7] / 1e6;   // sec/sec
    }

    else
    {
        sat_pos[0] = 0;
        sat_pos[1] = 0;
        sat_pos[2] = 0;
        sat_pos[3] = 0;

        sat_pos[4] = 0;
        sat_pos[5] = 0;
        sat_pos[6] = 0;
        sat_pos[7] = 0;
    }

    return retval;    
}

int main()
{
    //s_SV_Pos sat_pos;
    int retval = 0;
    char infilename[200] = "/mnt/c/Users/mlav736/Documents/GitHub/SPOC-Rongowai-L1/sxs/lib/igr21384.sp3";  //sp3-d filename is defined here

    unsigned int prn = 32;
    unsigned int week = 2138;
    double TOW = 345618.499262;

    retval = GetSVInfo(prn, week, TOW, sat_pos, infilename);

    //printf(PosVel);
    printf("\nGPS Position XYZ ECEF = %lf,%lf,%lf \n", sat_pos[0], sat_pos[1], sat_pos[2]);
    printf("GPS Velocity XYZ ECEF = %lf,%lf,%lf \n", sat_pos[4], sat_pos[5], sat_pos[6]);
    printf("GPS clock bias and drift = %12.11f,%12.11f \n", sat_pos[3], sat_pos[7]);
    return 0;
}