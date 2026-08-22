import { useRouter } from 'expo-router';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

export default function HomeScreen() {
  const router = useRouter();
  const theme = useTheme();

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.background }]}>
      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Branding */}
        <View style={styles.header}>
          <View style={styles.brandRow}>
            <View style={styles.logoBadge}>
              <Text style={styles.logoBadgeText}>🩺</Text>
            </View>
            <View>
              <ThemedText style={styles.brandTitle}>Nivora</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                AI Digital Biomarkers & Neurological Screening
              </ThemedText>
            </View>
          </View>
        </View>

        {/* Primary Featured Screening Card */}
        <ThemedView type="backgroundElement" style={styles.featuredCard}>
          <View style={styles.badgeRow}>
            <View style={styles.liveTag}>
              <Text style={styles.liveTagText}>ACTIVE CLINICAL MODULE</Text>
            </View>
            <ThemedText type="small" themeColor="textSecondary">
              10s Voice Test
            </ThemedText>
          </View>

          <ThemedText style={styles.featuredTitle}>Parkinson&apos;s Voice Screening</ThemedText>
          <ThemedText style={styles.featuredDesc} themeColor="textSecondary">
            Early detection of hypokinetic dysarthria and vocal cord micro-tremors using sustained vowel
            phonation (/a/). Analyzes pitch stability, jitter, shimmer, and harmonics-to-noise ratio (HNR).
          </ThemedText>

          <View style={styles.metricHighlightsRow}>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>F0 Pitch</Text>
              <Text style={styles.metricPillVal}>Micro-variations</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Jitter %</Text>
              <Text style={styles.metricPillVal}>Perturbation</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>HNR</Text>
              <Text style={styles.metricPillVal}>Acoustic Purity</Text>
            </View>
          </View>

          <Pressable
            style={({ pressed }) => [styles.startBtn, pressed && styles.pressed]}
            onPress={() => router.push('/parkinsons')}
          >
            <Text style={styles.startBtnText}>Launch Voice Screening →</Text>
          </Pressable>
        </ThemedView>

        {/* Additional Screening Modules Section */}
        <View style={styles.sectionHeader}>
          <ThemedText style={styles.sectionTitle}>Screening Modules</ThemedText>
          <ThemedText type="small" themeColor="textSecondary">
            Comprehensive neurological assessment suite
          </ThemedText>
        </View>

        <View style={styles.modulesGrid}>
          {/* Voice Assessment */}
          <Pressable
            style={({ pressed }) => [styles.moduleCard, pressed && styles.pressed]}
            onPress={() => router.push('/parkinsons')}
          >
            <View style={styles.moduleIconBox}>
              <Text style={styles.moduleIcon}>🎙️</Text>
            </View>
            <View style={styles.moduleContent}>
              <View style={styles.moduleTitleRow}>
                <ThemedText style={styles.moduleTitle}>Vocal Phonation (PD)</ThemedText>
                <Text style={styles.readyPill}>Ready</Text>
              </View>
              <ThemedText type="small" themeColor="textSecondary">
                Acoustic analysis using 10-second prolonged vowel test.
              </ThemedText>
            </View>
          </Pressable>

          {/* Motor Tapping Test */}
          <View style={[styles.moduleCard, styles.moduleCardMuted]}>
            <View style={[styles.moduleIconBox, styles.iconBoxMuted]}>
              <Text style={styles.moduleIcon}>👆</Text>
            </View>
            <View style={styles.moduleContent}>
              <View style={styles.moduleTitleRow}>
                <ThemedText style={styles.moduleTitle}>Motor Rhythm Tapping</ThemedText>
                <Text style={styles.comingSoonPill}>Upcoming</Text>
              </View>
              <ThemedText type="small" themeColor="textSecondary">
                Finger-tapping cadence and motor bradykinesia tracking.
              </ThemedText>
            </View>
          </View>

          {/* Postural Tremor */}
          <View style={[styles.moduleCard, styles.moduleCardMuted]}>
            <View style={[styles.moduleIconBox, styles.iconBoxMuted]}>
              <Text style={styles.moduleIcon}>📱</Text>
            </View>
            <View style={styles.moduleContent}>
              <View style={styles.moduleTitleRow}>
                <ThemedText style={styles.moduleTitle}>Kinematic Tremor Index</ThemedText>
                <Text style={styles.comingSoonPill}>Upcoming</Text>
              </View>
              <ThemedText type="small" themeColor="textSecondary">
                Accelerometer-based resting and postural tremor assessment.
              </ThemedText>
            </View>
          </View>

          {/* Research & Dataset Explorer */}
          <Pressable
            style={({ pressed }) => [styles.moduleCard, styles.moduleCardResearch, pressed && styles.pressed]}
            onPress={() => router.push('/explore')}
          >
            <View style={[styles.moduleIconBox, styles.iconBoxResearch]}>
              <Text style={styles.moduleIcon}>📊</Text>
            </View>
            <View style={styles.moduleContent}>
              <View style={styles.moduleTitleRow}>
                <ThemedText style={styles.moduleTitle}>UCI Dataset & Biomarkers</ThemedText>
                <Text style={styles.researchPill}>Research</Text>
              </View>
              <ThemedText type="small" themeColor="textSecondary">
                756 patient samples, 755 acoustic features, and 0.909 ROC AUC model.
              </ThemedText>
            </View>
          </Pressable>
        </View>

        {/* Clinical Disclaimer */}
        <View style={styles.footerNote}>
          <ThemedText type="small" themeColor="textSecondary" style={styles.footerText}>
            Medical Screening AI provides digital health screening tools based on research datasets (UCI PD Speech Features). This tool is intended for exploratory analysis and does not replace medical diagnosis.
          </ThemedText>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  scrollContainer: {
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.four,
    paddingBottom: BottomTabInset + Spacing.four,
    maxWidth: MaxContentWidth,
    alignSelf: 'center',
    width: '100%',
    gap: Spacing.four,
  },
  header: {
    marginBottom: Spacing.two,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  logoBadge: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#E0F2FE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoBadgeText: {
    fontSize: 22,
  },
  brandTitle: {
    fontSize: 20,
    fontWeight: '700',
    letterSpacing: -0.3,
  },
  featuredCard: {
    borderRadius: 20,
    padding: Spacing.four,
    borderWidth: 1,
    borderColor: '#BAE6FD',
    gap: Spacing.three,
    backgroundColor: '#F0F9FF',
  },
  badgeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  liveTag: {
    backgroundColor: '#0284C7',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  liveTagText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  featuredTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#0369A1',
  },
  featuredDesc: {
    fontSize: 14,
    lineHeight: 21,
    color: '#334155',
  },
  metricHighlightsRow: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  metricPill: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  metricPillLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: '#0284C7',
  },
  metricPillVal: {
    fontSize: 12,
    color: '#475569',
  },
  startBtn: {
    backgroundColor: '#0284C7',
    paddingVertical: 15,
    borderRadius: 14,
    alignItems: 'center',
    shadowColor: '#0284C7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 3,
  },
  startBtnText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  sectionHeader: {
    marginTop: Spacing.two,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 2,
  },
  modulesGrid: {
    gap: 12,
  },
  moduleCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.three,
    borderRadius: 14,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E2E8F0',
    gap: 12,
  },
  moduleCardMuted: {
    opacity: 0.75,
    backgroundColor: '#F8FAFC',
  },
  moduleIconBox: {
    width: 44,
    height: 44,
    borderRadius: 10,
    backgroundColor: '#E0F2FE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBoxMuted: {
    backgroundColor: '#F1F5F9',
  },
  moduleIcon: {
    fontSize: 20,
  },
  moduleContent: {
    flex: 1,
  },
  moduleTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  moduleTitle: {
    fontSize: 15,
    fontWeight: '600',
  },
  readyPill: {
    fontSize: 11,
    fontWeight: '700',
    color: '#059669',
    backgroundColor: '#D1FAE5',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  comingSoonPill: {
    fontSize: 11,
    fontWeight: '600',
    color: '#64748B',
    backgroundColor: '#E2E8F0',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  moduleCardResearch: {
    borderColor: '#BAE6FD',
    backgroundColor: '#F0F9FF',
  },
  iconBoxResearch: {
    backgroundColor: '#E0F2FE',
  },
  researchPill: {
    fontSize: 11,
    fontWeight: '700',
    color: '#0284C7',
    backgroundColor: '#E0F2FE',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  footerNote: {
    padding: Spacing.three,
    backgroundColor: '#F8FAFC',
    borderRadius: 10,
    marginTop: Spacing.two,
  },
  footerText: {
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
  },
  pressed: {
    opacity: 0.8,
    transform: [{ scale: 0.98 }],
  },
});
