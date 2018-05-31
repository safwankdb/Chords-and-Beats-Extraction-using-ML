function [CRP, sideinfo] = extract_CRP(directory,file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create Parameter

parameter.useResampling = 1;
parameter.destSamplerate = 22050;
parameter.convertToMono = 1;
parameter.monoConvertMode = 'downmix';
parameter.message = 0;
parameter.vis = 0;
parameter.save = 0;
parameter.saveDir = ['',directory];
parameter.saveFilename = file;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Convert to audio

[audio, sideinfo] = wav_to_audio('',directory,file,parameter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Extract shiftFB

shiftFB=estimateTuning(audio);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Update Parameter

clear parameter
parameter.winLenSTMSP = 4410;
parameter.fs = sideinfo.wav.fs;
parameter.save = 1;
parameter.saveDir = 'data_feature/';
parameter.saveFilename = file;
parameter.shiftFB = shiftFB;
parameter.saveAsTuned = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Extract Pitch features

[pitch,sideinfo] = audio_to_pitch_via_FB(audio,parameter,sideinfo);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Update Parameter

clear parameter
parameter.coeffsToKeep = [55:120];
parameter.applyLogCompr = 1;
parameter.factorLogCompr = 1000;
parameter.addTermLogCompr = 1;
parameter.normP = 2;
parameter.winLenSmooth = 1;
parameter.downsampSmooth = 1;
parameter.normThresh = 10^-6;
parameter.inputFeatureRate = 0;
parameter.save = 0;
parameter.saveDir = '';
parameter.saveFilename = '';
parameter.visualize = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Extract CRP

[CRP, sideinfo]= pitch_to_CRP(pitch,parameter,sideinfo);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualization of CRP chromagram 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parameter.featureRate = sideinfo.CRP.featureRate;
parameter.xlabel = 'Time [Seconds]';
parameter.title = 'CRP chromagram';
visualizeCRP(CRP,parameter);
end