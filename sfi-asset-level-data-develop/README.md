# sfi-asset-level-data Repository

## Contributing Code

This repository has been configured for GitFlow processing, which makes it easier for contributors to commit and push works in progress without accidently overwriting critical code components. General guidelines for use:

1. Kim and Courtney will manage the <code>develop</code> and <code>master</code> branches.
2. For active development, create a feature branch called <code>feature/[your branch name]</code>, and do your work here.
3. Create features branches off of <code>develop</code>.
4. When your code is at a satifying level of "doneness", we'll merge your feature branch into <code>develop</code>.

## Repository File Organization

This repository includes experimental and final versions of code for Phase 1 and Phase 2 of the ALD project. For Phase 1, analysts used multiple platforms and GitHub respositories to individually track their code. For Phase 2, we are migrating work to this single GitHub repository. The directory structure for this repository is as follows:

1. <code>phase-1-final</code> includes all final code copied from separate repositories, at the time it was delivered to CIFF. Note that these are simple copies, and do not maintain the history of those repositories:
    1. <code>s22s--sfi-asset-level-data</code>: Steel plant localization and web scraping.
    2. <code>sreece101--CIFF-ALD</code>: Macro- and micro-localization for cement.
    3. <code>chris010970--ald</code>: Data preparation and cement asset attributes estimation.
2. <code>src/main</code> is where we will put all new code developed in Phase 2. This directory including subdirectories:
    1. <code>resources</code> is where we can store small data sets that the team frequently needs to access. Do not store large files, such as images, here.
    2. <code>jupyter</code> is where we will keep \*.ipynb files. To keep the file size down, it's recommended to clear the outputs before committing these files.
    3. <code>[TBD]</code> we can create other subfolders as needed for different types.

    
## Storing Large Files

Large files, such as GeoTIFFs or even large number of JPGs, generally are not appropriate to store/track in GitHub. To work with such files in EarthAI Notebook, we will make use of a few storage options:

1. AWS S3: The ALD project has a shared, private S3 bucket to store any type of file on. This is best suited for large, static files, to be read in only once from a notebook. The example notebook here demonstrates how to read from and upload to objects in S3: https://github.com/s22s/sfi-asset-level-data/blob/develop/src/main/jupyter/aws-access.ipynb. All SFI analysts have access to this bucket, and can access these resources simultaneously.
2. Local storage in EarthAI Notebook: This is what is shown in the left sidebar. Within a terminal or notebook, the home path to this local storage is "/home/jovyan". This is a good place to store your local copy of this GitHub repository (using the built-in Git integration tools). It's a good place to store relatively small files while you're iterating on something new. Anything stored here will persist from session to session (i.e. when you shut down and relaucnch a server). However, files that you upload to the local storage are not shared among your colleagues.
3. Fast local storage in EarthAI Notebook: The fast local storage is an attached SSD, making read/write faster. Within a terminal or notebook, you can access it by "/scratch". As with the regular local storage, what you put in fast local storage is not shared among your colleagues. However, it also does NOT persist between sessions. Fast local storage is best to read/create image chips to be used for deep learning. An example of its use is shown here: https://github.com/s22s/sfi-asset-level-data/blob/develop/src/main/jupyter/cement-micro-location-model/cement_unet_vgg16_phase1.ipynb.
