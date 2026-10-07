# Study: Agentic Debt final — 110-chain base and proposed 121-chain expansion

**Date:** 2026-10-06
**Status:** Closed
**Question:** Which chains belong in the low-mean subset and its high-variance and negative-mean expansion?
**Deployment scope:** The active v7 manifest contains the original 110 chains. The proposed expansion below has not been published or deployed.
**Runs and data used:** GLM final-v6 `acuadron-agentic-debt-final-v6-t66h9`; Opus `acuadron-adfinal-opus48-high-x8-1006`.
**Code SHAs:** No trainer code change. Dataset manifest commit `249fc194d84aef7ac8047d87b0b07bd558ec68369bbc77854e40e7fbdfa17f2b`.
**Workflow ids:** `acuadron-agentic-debt-final-v6-t66h9`.

## Method

Use the analysis snapshot from 2026-10-06, not future results from the active run.
Exclude trials with nonnumeric rewards or any recorded trial or step exception.
Join the two runs by the canonical chain ID in each trial result.
For each chain, use the latest available GLM weight version with at least eight clean trials.
If no GLM version qualifies, use Opus results with at least eight clean trials.
Compute the mean and population standard deviation across the selected source trials for each chain.
Keep candidates with SD strictly greater than 0.15. Sort by ascending mean, then canonical chain ID. Select the first 110.
Apply no minimum mean filter and no additional Opus difficulty filter to GLM-measured chains.
Weight each selected chain equally for the aggregate estimate.

## Original 110-chain results

| Metric | Value |
| --- | ---: |
| Eligible chains | 128 |
| Selected chains | 110 |
| GLM-measured chains | 106 |
| Opus-proxy chains | 4 |
| Equal-chain mean estimate | 0.515007 |
| Opus mean on the selected subset | 0.649762 |
| Historical pooled GLM mean plus Opus proxies | 0.628418 |
| Highest selected mean | 0.833333 |

## Original 110 selected chains

Rows follow selection rank. Mean and SD use the source in the last column.
`wvNNNN` denotes the GLM weight-version label. `Opus proxy` denotes a chain without sufficient GLM evidence.
Statistics below are rounded; selection uses full precision.

| Rank | Canonical chain ID | Mean | SD | Clean attempts | Source |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `realdiff_googlefonts_ufomerge_s0` | -0.1875 | 0.2421 | 8 | wv0005 |
| 2 | `realdiff_iamjackg_md2cf_s0` | -0.0590 | 0.3063 | 8 | wv0005 |
| 3 | `realdiff_ipnet-mesh_meshcore-mqtt_s0` | 0.0020 | 0.5032 | 24 | wv0001 |
| 4 | `realdiff_scverse_muon_s0` | 0.0329 | 0.2630 | 8 | wv0004 |
| 5 | `realdiff_HTTP-APIs_hydra-python-core_s0` | 0.0833 | 0.1667 | 8 | wv0007 |
| 6 | `realdiff_WhaleOps_air2phin_s0` | 0.0833 | 0.2764 | 24 | wv0001 |
| 7 | `realdiff_dahlia-lib_dahlia_s0` | 0.1033 | 0.2826 | 8 | wv0004 |
| 8 | `realdiff_lzakharov_csv2md_s0` | 0.1111 | 0.2940 | 8 | wv0004 |
| 9 | `realdiff_raulgomis_semversioner_s1` | 0.1250 | 0.3307 | 8 | wv0008 |
| 10 | `realdiff_testingautomated-usi_uncertainty-wizard_s1` | 0.1250 | 0.2165 | 8 | wv0007 |
| 11 | `realdiff_gristlabs_asttokens_s17` | 0.1493 | 0.3557 | 8 | wv0002 |
| 12 | `realdiff_jsfehler_flake8-multiline-containers_s1` | 0.1623 | 0.4140 | 8 | wv0003 |
| 13 | `realdiff_The-Strategy-Unit_nhp_model_s0` | 0.1654 | 0.3699 | 24 | wv0001 |
| 14 | `realdiff_econchick_interrogate_s0` | 0.1694 | 0.2654 | 8 | wv0002 |
| 15 | `realdiff_joegasewicz_bobtail_s0` | 0.1970 | 0.4326 | 16 | wv0001 |
| 16 | `realdiff_altwalker_altwalker_s0` | 0.2174 | 0.2883 | 8 | wv0005 |
| 17 | `realdiff_lidatong_dataclasses-json_s0` | 0.2307 | 0.4033 | 24 | wv0001 |
| 18 | `realdiff_rl-institut_smooth_s0` | 0.2500 | 0.3536 | 8 | wv0005 |
| 19 | `realdiff_slarse_labelbot_s0` | 0.2526 | 0.6761 | 8 | wv0005 |
| 20 | `realdiff_adafruit_circup_s6` | 0.2615 | 0.3507 | 8 | wv0006 |
| 21 | `realdiff_alan-turing-institute_sqlsynthgen_s0` | 0.2636 | 0.4429 | 8 | wv0001 |
| 22 | `realdiff_CircleCI-Public_aws-ecs-orb_s0` | 0.2747 | 0.4230 | 8 | wv0003 |
| 23 | `realdiff_scottstanie_sentineleof_s7` | 0.2750 | 0.2613 | 8 | wv0006 |
| 24 | `realdiff_ansys_ansys-templates_s0` | 0.2986 | 0.2386 | 8 | wv0006 |
| 25 | `realdiff_SUSE-Enceladus_img-proof_s8` | 0.3047 | 0.4815 | 16 | wv0001 |
| 26 | `realdiff_jbusecke_xmovie_s0` | 0.3320 | 0.4491 | 8 | wv0002 |
| 27 | `realdiff_tarioch_xirr_s1` | 0.3438 | 0.3292 | 8 | wv0006 |
| 28 | `realdiff_tonybaloney_wily_s0` | 0.3458 | 0.4683 | 8 | wv0002 |
| 29 | `realdiff_khornberg_octokit.py_s0` | 0.3602 | 0.4375 | 8 | wv0004 |
| 30 | `realdiff_gristlabs_asttokens_s13` | 0.3611 | 0.4148 | 8 | wv0004 |
| 31 | `realdiff_scikit-tda_cechmate_s0` | 0.3694 | 0.3996 | 24 | wv0001 |
| 32 | `realdiff_gristlabs_asttokens_s1` | 0.3757 | 0.3941 | 24 | wv0001 |
| 33 | `realdiff_ros-infrastructure_rosdoc2_s21` | 0.3889 | 0.2152 | 8 | wv0006 |
| 34 | `realdiff_CrowdStrike_caracara_s0` | 0.3964 | 0.4279 | 8 | wv0003 |
| 35 | `realdiff_stigok_ruterstop_s0` | 0.3979 | 0.4275 | 8 | wv0003 |
| 36 | `realdiff_JudeWells_chainsaw_s0` | 0.4130 | 0.3897 | 23 | wv0001 |
| 37 | `realdiff_UM-ARM-Lab_pytorch_kinematics_s0` | 0.4375 | 0.1654 | 8 | wv0005 |
| 38 | `realdiff_adamchainz_django-rich_s0` | 0.4375 | 0.3248 | 8 | wv0006 |
| 39 | `realdiff_canonical_operator-libs-linux_s0` | 0.4438 | 0.4767 | 8 | Opus proxy |
| 40 | `realdiff_blingenf_copydetect_s0` | 0.4511 | 0.3266 | 16 | wv0004 |
| 41 | `realdiff_LKI_chinese-calendar_s0` | 0.4609 | 0.5475 | 8 | Opus proxy |
| 42 | `realdiff_astronomer_astro-provider-anyscale_s0` | 0.4710 | 0.2560 | 8 | wv0007 |
| 43 | `realdiff_strictdoc-project_reqif_s0` | 0.4767 | 0.1808 | 8 | wv0003 |
| 44 | `realdiff_jaxvanyang_dotbackup_s0` | 0.4792 | 0.4340 | 8 | wv0006 |
| 45 | `realdiff_Continvvm_continuum_s0` | 0.4864 | 0.5046 | 8 | wv0001 |
| 46 | `realdiff_aws_aws-iot-device-sdk-python-v2_s0` | 0.5139 | 0.3765 | 8 | wv0005 |
| 47 | `realdiff_sensein_cmixf_s0` | 0.5208 | 0.4819 | 8 | wv0006 |
| 48 | `realdiff_PyCQA_mccabe_s0` | 0.5312 | 0.4911 | 24 | wv0001 |
| 49 | `realdiff_Quantco_slim-trees_s0` | 0.5333 | 0.2309 | 8 | wv0006 |
| 50 | `realdiff_dennis6p_adaptive-cards-py_s0` | 0.5354 | 0.2152 | 8 | wv0004 |
| 51 | `realdiff_canonical_data-science-stack_s0` | 0.5423 | 0.3458 | 16 | wv0001 |
| 52 | `realdiff_Almas-Ali_envist_s0` | 0.5441 | 0.3039 | 8 | wv0007 |
| 53 | `realdiff_diku-dk_RAINBOW_s1` | 0.5500 | 0.3918 | 8 | wv0006 |
| 54 | `realdiff_fonttools_ttfautohint-py_s0` | 0.5764 | 0.2541 | 8 | wv0006 |
| 55 | `realdiff_sensepost_objection_s0` | 0.5806 | 0.1938 | 8 | wv0005 |
| 56 | `realdiff_python_miss-islington_s0` | 0.5891 | 0.2524 | 8 | wv0006 |
| 57 | `realdiff_pytroll_trollmoves_s0` | 0.5948 | 0.3531 | 8 | wv0005 |
| 58 | `realdiff_ApptuitAI_apptuit-py_s0` | 0.5990 | 0.2638 | 8 | wv0003 |
| 59 | `realdiff_fnproject_fdk-python_s0` | 0.6042 | 0.3674 | 8 | wv0005 |
| 60 | `realdiff_conda_conda-package-handling_s0` | 0.6045 | 0.2669 | 8 | wv0004 |
| 61 | `realdiff_acsone_setuptools-odoo_s0` | 0.6049 | 0.3720 | 8 | wv0005 |
| 62 | `realdiff_LabSid-USP_RUBEM_s3` | 0.6086 | 0.1532 | 8 | wv0005 |
| 63 | `realdiff_melexis_sphinx-traceability-extension_s0` | 0.6129 | 0.4758 | 8 | wv0001 |
| 64 | `realdiff_banesullivan_scooby_s0` | 0.6161 | 0.3879 | 8 | wv0003 |
| 65 | `realdiff_scrapinghub_price-parser_s0` | 0.6188 | 0.4795 | 8 | wv0001 |
| 66 | `realdiff_latchfield_vulcan-core_s0` | 0.6299 | 0.2471 | 24 | wv0001 |
| 67 | `realdiff_Gardene-el_Coze2JianYing_s4` | 0.6417 | 0.4630 | 8 | wv0007 |
| 68 | `realdiff_joergbuchwald_ogs6py_s0` | 0.6457 | 0.2595 | 24 | wv0001 |
| 69 | `realdiff_adamchainz_flake8-logging_s0` | 0.6472 | 0.3929 | 8 | wv0002 |
| 70 | `realdiff_capeprivacy_cape-dataframes_s11` | 0.6508 | 0.4094 | 11 | wv0001 |
| 71 | `realdiff_bytedance_trae-agent_s0` | 0.6557 | 0.5127 | 16 | wv0001 |
| 72 | `realdiff_houfu_redlines_s0` | 0.6562 | 0.2480 | 8 | wv0005 |
| 73 | `realdiff_wagnerdelima_drf-social-oauth2_s0` | 0.6667 | 0.2887 | 8 | wv0005 |
| 74 | `realdiff_vb64_telemulator3_s0` | 0.6670 | 0.2756 | 16 | wv0004 |
| 75 | `realdiff_patman15_BMS_BLE-HA_s0` | 0.6746 | 0.1662 | 16 | wv0001 |
| 76 | `realdiff_houfu_redlines_s6` | 0.6753 | 0.3330 | 16 | wv0004 |
| 77 | `realdiff_drvinceknight_nbchkr_s0` | 0.6808 | 0.1711 | 8 | wv0004 |
| 78 | `realdiff_jaylinski_kodi-addon-soundcloud_s0` | 0.6821 | 0.2304 | 8 | wv0006 |
| 79 | `realdiff_prefab-cloud_prefab-cloud-python_s16` | 0.6823 | 0.2854 | 8 | wv0006 |
| 80 | `realdiff_modernatx_seqlike_s0` | 0.6831 | 0.4480 | 8 | wv0006 |
| 81 | `realdiff_soar-telescope_goodman_focus_s0` | 0.6875 | 0.2841 | 8 | wv0006 |
| 82 | `realdiff_cylc_cylc-rose_s0` | 0.6945 | 0.4474 | 24 | wv0001 |
| 83 | `realdiff_SpamScope_mail-parser_s0` | 0.7051 | 0.4635 | 8 | Opus proxy |
| 84 | `realdiff_PrefectHQ_prefect-monte-carlo_s0` | 0.7076 | 0.2726 | 8 | wv0006 |
| 85 | `realdiff_art1415926535_graphene-sqlalchemy-filter_s0` | 0.7118 | 0.2958 | 8 | Opus proxy |
| 86 | `realdiff_4n4nd_prometheus-api-client-python_s0` | 0.7133 | 0.3355 | 8 | wv0006 |
| 87 | `realdiff_NowanIlfideme_pydantic-kedro_s0` | 0.7167 | 0.2925 | 8 | wv0004 |
| 88 | `realdiff_canonical_awsmp_s0` | 0.7167 | 0.4173 | 8 | wv0004 |
| 89 | `realdiff_xarray-contrib_xproj_s0` | 0.7208 | 0.4229 | 8 | wv0003 |
| 90 | `realdiff_openpodcast_pipelines_s0` | 0.7273 | 0.3087 | 8 | wv0003 |
| 91 | `realdiff_bluesky_event-model_s0` | 0.7331 | 0.2984 | 8 | wv0003 |
| 92 | `realdiff_cantools_cantools_s0` | 0.7454 | 0.3698 | 8 | wv0001 |
| 93 | `realdiff_vietspeak_VietSpeakOJ_s0` | 0.7634 | 0.1546 | 8 | wv0005 |
| 94 | `realdiff_aiidateam_qe-tools_s7` | 0.7647 | 0.2380 | 8 | wv0005 |
| 95 | `realdiff_kpetremann_mqtt-exporter_s0` | 0.7775 | 0.5127 | 8 | wv0004 |
| 96 | `realdiff_SAP_swagger-plugin-for-sphinx_s0` | 0.7775 | 0.2727 | 24 | wv0001 |
| 97 | `realdiff_kangasta_fdbk_s0` | 0.7857 | 0.3571 | 8 | wv0003 |
| 98 | `realdiff_SUSE-Enceladus_img-proof_s0` | 0.7983 | 0.3231 | 8 | wv0004 |
| 99 | `realdiff_thesimj_envyaml_s0` | 0.8000 | 0.3097 | 8 | wv0005 |
| 100 | `realdiff_OpenTrafficCam_OTCamera_s0` | 0.8101 | 0.1689 | 8 | wv0006 |
| 101 | `realdiff_davideganna_DataMorphers_s0` | 0.8112 | 0.4360 | 8 | wv0007 |
| 102 | `realdiff_amaranth-lang_amaranth-soc_s0` | 0.8145 | 0.3282 | 24 | wv0001 |
| 103 | `realdiff_mozilla-services_requests-hawk_s0` | 0.8199 | 0.1696 | 8 | wv0007 |
| 104 | `realdiff_craigthomas_Chip8Assembler_s0` | 0.8250 | 0.3711 | 24 | wv0001 |
| 105 | `realdiff_pypolestar_pypolestar_s0` | 0.8264 | 0.3284 | 8 | wv0006 |
| 106 | `realdiff_BlueBrain_hpc-coding-conventions_s0` | 0.8281 | 0.3362 | 8 | wv0004 |
| 107 | `realdiff_tidewave-ai_tidewave_python_s27` | 0.8302 | 0.2205 | 8 | wv0004 |
| 108 | `realdiff_laminlabs_nbproject_s0` | 0.8320 | 0.3216 | 8 | wv0006 |
| 109 | `realdiff_mkdocs_mkdocs-click_s0` | 0.8329 | 0.3536 | 16 | wv0001 |
| 110 | `realdiff_vaidik_commentjson_s0` | 0.8333 | 0.1863 | 8 | wv0006 |

## Proposed additions: eight high-variance chains

Retain the original 110 chains. Add the eight omitted candidates with the highest observed SD, all above 0.25.
Their high means excluded them from the original cutoff, but their within-group reward differences provide potential GRPO signal.
Rows below continue the inventory numbering; they do not represent a new mean-based rank.

| Number | Canonical chain ID | Mean | SD | Clean attempts | Source |
| ---: | --- | ---: | ---: | ---: | --- |
| 111 | `realdiff_evanjd_python-logi-circle_s0` | 0.8716 | 0.3379 | 8 | wv0003 |
| 112 | `realdiff_openfoodfacts_facets-knowledge-panels_s0` | 0.8667 | 0.3288 | 24 | wv0001 |
| 113 | `realdiff_pmacg_py-sgtl_s0` | 0.8685 | 0.3283 | 8 | wv0001 |
| 114 | `realdiff_icecube_skyllh_s0` | 0.8349 | 0.3269 | 16 | wv0001 |
| 115 | `realdiff_lisad_phaser_s0` | 0.8617 | 0.3259 | 8 | wv0001 |
| 116 | `realdiff_didix21_mdutils_s0` | 0.8948 | 0.3136 | 24 | wv0001 |
| 117 | `realdiff_ArkEcosystem_python-crypto_s0` | 0.8612 | 0.2902 | 24 | wv0001 |
| 118 | `realdiff_zheller_flake8-quotes_s0` | 0.8977 | 0.2716 | 24 | wv0001 |

## Proposed additions: three negative-mean chains

Add all omitted chains with negative estimated means, as requested. This rule overrides the original SD threshold for these three chains.
The original 110 already include the other two negative-mean chains: `realdiff_googlefonts_ufomerge_s0` and `realdiff_iamjackg_md2cf_s0`.
These additions provide hard-case coverage. Their low variance does not establish strong GRPO signal.

| Number | Canonical chain ID | Mean | SD | Clean attempts | Source |
| ---: | --- | ---: | ---: | ---: | --- |
| 119 | `realdiff_dw-0_kiauh_s0` | -0.0213 | 0.0690 | 24 | wv0001 |
| 120 | `realdiff_eWaterCycle_era5cli_s0` | -0.0399 | 0.0535 | 8 | Opus proxy |
| 121 | `realdiff_robin900_gspread-dataframe_s0` | -0.0050 | 0.0087 | 8 | wv0002 |

## Expanded selection summary

| Subset | Chains | Equal-chain mean estimate |
| --- | ---: | ---: |
| Original selection | 110 | 0.515007 |
| Original plus eight high-variance chains | 118 | 0.539050 |
| Original plus eight high-variance and three negative-mean chains | 121 | 0.525138 |

The 121-chain proposal contains 116 GLM-measured chains and five Opus proxies.
Negative-mean additions refer to chain means, not every chain with an individual negative attempt.

## Verdict

The proposed next subset contains 121 chains with an observed equal-chain mean estimate of approximately 0.525.
This estimate combines different GLM weight versions and five Opus proxies. It is not a forecast for one frozen checkpoint.
The active training run remains on the original 110-chain manifest.

## Caveats and open items

- Selection and estimates use the same observations; a fresh evaluation is necessary to validate the next-run mean.
- The estimate concerns equal-chain rewards, not segment-weighted `raw_reward` or the post-filter training mean.
- Nonzero reward variance does not establish useful gradient signal or verifier correctness.
- Opus proxies do not establish GLM difficulty on those five chains in the expanded proposal.
- This document supersedes the earlier 37-chain proposal for the selected subset.

## Actions taken

- Preserved the original 110 selected chain IDs and their source statistics.
- Added eight high-variance candidates and three negative-mean candidates for the proposed 121-chain subset.
- This document update did not publish a new manifest, change the active training configuration, restart a run, or create a Git commit.

## Sources

- Opus job: https://harbor.pdx.beta.arena.agif.amazon.dev/jobs/acuadron-adfinal-opus48-high-x8-1006
- Opus results: `s3://arena-scratch-beta-pdx-us-west-2/harbor-jobs/acuadron-adfinal-opus48-high-x8-1006/.trials-index.json`
- GLM trials: `s3://arena-scratch-prod-bom-ap-south-1/acuadron/harbor-training/acuadron-agentic-debt-final-v6/`
- Local analysis snapshot: `/tmp/glm53f/opus1006-map/joined.json`
- Analysis snapshot SHA-256: `5a927053348e20d9ab09c7ed30784e1a8e4daee30bc0b296c08c7f38dca3b572`
